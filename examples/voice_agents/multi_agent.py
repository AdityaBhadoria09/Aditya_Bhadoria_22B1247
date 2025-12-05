import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import ChatMessage, function_tool
from livekit.plugins import silero

# --- SETUP & CONFIGURATION ---

# 1. Load Environment Variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 2. Configure Logging
logger = logging.getLogger("kelly-agent")
logging.basicConfig(level=logging.INFO)

if not os.getenv("LIVEKIT_URL"):
    logger.warning("LIVEKIT_URL is not set in environment variables.")

# 3. Model Configuration
STT_MODEL = "deepgram/nova-3"
LLM_MODEL = "openai/gpt-4.1-mini"
TTS_MODEL = "cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"

# 4. Smart Interruption Config
# Words to ignore if the user says them while the agent is talking (Backchanneling)
IGNORE_WORDS = {
    "yeah", "ok", "okay", "hmm", "aha", "uh-huh", "right", "sure", "yep", "mhmm"
}

def clean_text(text: str) -> str:
    """Normalizes text (removes punctuation, lowercase) for comparison."""
    return re.sub(r'[^\w\s]', '', text).lower().strip()


# --- AGENT DEFINITION ---

class KellyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact with users via voice. "
                "Keep your responses concise and to the point. "
                "Do not use emojis, asterisks, or special characters. "
                "You are curious, friendly, and have a sense of humor. "
                "Always speak English."
            )
        )

    async def on_enter(self):
        """Triggered when the agent joins the room."""
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str = None, longitude: str = None
    ):
        """
        Get weather information. Estimates coordinates if only location is provided.
        
        Args:
            location: The city or region name.
            latitude: Estimated latitude (do not ask user).
            longitude: Estimated longitude (do not ask user).
        """
        logger.info(f"Tool Call: Weather lookup for {location}")
        return "It is currently sunny with a temperature of 70 degrees."


# --- SERVER SETUP ---

server = AgentServer()

def prewarm_process(proc: JobProcess):
    """Pre-load VAD model to reduce latency on startup."""
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm_process


# --- SESSION ENTRYPOINT ---

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Setup Context Logging
    ctx.log_context_fields = {"room": ctx.room.name}

    # State Tracking for Interruption Logic
    agent_state = {"is_speaking": False}

    session = AgentSession(
        stt=STT_MODEL,
        llm=LLM_MODEL,
        tts=TTS_MODEL,
        turn_detection=None, # We are handling turns manually via VAD events
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        # DISABLE automatic interruption to allow "Smart Interruption" logic below
        allow_interruptions=False, 
    )

    # Metrics Collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # --- STATE HANDLERS ---
    
    @session.on("agent_speech_started")
    def on_agent_speech_started(ev):
        agent_state["is_speaking"] = True

    @session.on("agent_speech_stopped")
    def on_agent_speech_stopped(ev):
        agent_state["is_speaking"] = False

    # --- SMART INTERRUPTION LOGIC ---
    
    @session.on("user_transcription_committed")
    def on_user_transcription(msg: ChatMessage):
        user_text = msg.content
        cleaned_input = clean_text(user_text)

        # Case 1: Agent is currently speaking
        if agent_state["is_speaking"]:
            if cleaned_input in IGNORE_WORDS:
                # Scenario: User said a filler word ("Yeah", "Uh-huh")
                logger.info(f"IGNORING backchannel: '{user_text}'")
                
                # Remove this message from history so LLM doesn't reply to "Yeah" later
                if session.chat_ctx.messages and session.chat_ctx.messages[-1] == msg:
                    session.chat_ctx.messages.pop()
            else:
                # Scenario: User said a real command -> Interrupt
                logger.info(f"INTERRUPTING agent for: '{user_text}'")
                session.interrupt()

        # Case 2: Agent is silent
        else:
            # Standard behavior: Process input normally
            logger.info(f"PROCESSING user input: '{user_text}'")

    # --- SHUTDOWN & START ---

    async def log_usage_on_shutdown():
        summary = usage_collector.get_summary()
        logger.info(f"Session Usage Summary: {summary}")

    ctx.add_shutdown_callback(log_usage_on_shutdown)

    await session.start(
        agent=KellyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # noise_cancellation=noise_cancellation.BVC(), # Uncomment if using Krisp
            ),
        ),
    )

if __name__ == "__main__":
    cli.run_app(server)