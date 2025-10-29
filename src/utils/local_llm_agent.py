"""
Local LLM Agent - Complete replica of Modal implementation
Runs Qwen3-8B locally with all Modal features including:
- Enhanced game info extraction
- Phase-aware prompt generation
- Strategic move extraction
- Theory of Mind reasoning
"""
import torch
import re
import time
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMEngine:
    """Local LLM inference engine - exact replica of Modal implementation"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """
        Initialize local LLM engine matching Modal Labs configuration

        Args:
            model_name: Hugging Face model name (default: Qwen3-8B, same as Modal)
            device: Device to use (cuda/cpu)
            load_in_4bit: Use 4-bit quantization (recommended for RTX 4090)
            load_in_8bit: Use 8-bit quantization (same as Modal on A100)
        """
        start_time = time.time()
        logger.info("=== LOCAL LLM ENGINE INITIALIZATION (Modal-compatible) ===")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'none'}")

        self.device = device
        self.model_name = model_name

        # Load tokenizer (Modal line 97-103)
        logger.info("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization (Modal lines 106-114)
        logger.info("📥 Loading model (this may take 30-60 seconds)...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,  # Modal line 108
            "low_cpu_mem_usage": True,  # Modal line 112
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = "auto"
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True  # Modal line 111
            model_kwargs["device_map"] = "auto"  # Modal line 109
        else:
            model_kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.1f}s")

        # Create pipeline (Modal lines 117-129)
        logger.info("🔧 Creating inference pipeline...")
        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        logger.info(f"✅ Local LLM Engine ready!\n")

    def _parse_qwen_output(self, response: str) -> str:
        """Parse Qwen3-8B model output - handle thinking mode (Modal lines 356-389)"""
        if '</think>' in response:
            try:
                parts = response.split('</think>')
                if len(parts) > 1:
                    actual_response = parts[-1].strip()
                    if actual_response:
                        response = actual_response
                    else:
                        thinking_part = parts[0].replace('<think>', '').strip()
                        if thinking_part:
                            response = thinking_part
            except Exception as e:
                logger.warning(f"Error parsing thinking content: {e}")

        response = re.sub(r'</?[^>]*>', '', response).strip()
        response = re.sub(r'\s+', ' ', response).strip()

        if not response:
            return "I need to analyze the situation."

        return response

    # Remove _generate_enhanced_prompt - not needed anymore since we use StreamlinedMafiaAgent's prompts

    # Remove _extract_move - not needed anymore since we use StreamlinedMafiaAgent's extraction logic

    def generate(self, observation: str, system_prompt: str = "", phase: str = "discussion") -> Dict:
        """Generate response using EXACT same prompting as online agent (streamlined_mafia_agent.py)"""
        request_start = time.time()
        logger.info("=== LOCAL GENERATION REQUEST (Online Agent Compatible) ===")

        try:
            # Use system prompt directly from online agent - NO additional prompt engineering
            if not system_prompt:
                raise Exception("System prompt required from StreamlinedMafiaAgent")

            logger.info(f"Using StreamlinedMafiaAgent system prompt for phase: {phase}")

            # Simple message structure - exactly like online agent expects
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation}
            ]

            # Inference parameters matching online agent expectations
            if phase == "action":
                max_tokens = 10
                temperature = 0.7  # Match online agent
                top_k = 20
                top_p = 0.8
            else:
                max_tokens = 100
                temperature = 0.7  # Match online agent
                top_k = 20
                top_p = 0.8

            try:
                enable_thinking = phase != "action"
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
                )
            except:
                if len(messages) == 2:
                    formatted_prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}"
                else:
                    formatted_prompt = observation

            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,  # Match online agent
            )

            if isinstance(outputs, list) and len(outputs) > 0:
                raw_response = outputs[0]['generated_text'].strip()
            else:
                raw_response = ""

            raw_response = self._parse_qwen_output(raw_response)
            logger.info(f"Raw model output: '{raw_response}'")

            total_time = time.time() - request_start
            logger.info(f"Generation complete in {total_time:.3f}s")

            return {
                'response': raw_response,
                'reasoning': f"Phase: {phase}",
                'processing_time': total_time
            }

        except Exception as e:
            error_time = time.time() - request_start
            logger.error(f"Generation error: {e}")
            return {
                'response': "",
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': error_time
            }


class LocalStreamlinedMafiaAgent:
    """
    StreamlinedMafiaAgent adapted for local LLM inference
    Complete replica of Modal functionality running locally
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        """Initialize local agent with full Modal capabilities"""
        from .streamlined_mafia_agent import StreamlinedMafiaAgent

        # Initialize LLM engine with full Modal features
        self.llm_engine = LocalLLMEngine(
            model_name=model_name,
            device=device,
            load_in_4bit=load_in_4bit,
        )

        # Create base agent
        self.base_agent = StreamlinedMafiaAgent(modal_endpoint_url="http://localhost:8000")

        # Replace Modal call with local inference - using EXACT same system prompts as online agent
        def local_inference_replacement(user_prompt: str, phase: str = "discussion", timeout: int = 30) -> str:
            """
            Replace Modal HTTP call with local LLM inference.
            Uses EXACT same system prompts as StreamlinedMafiaAgent._call_modal_with_enhanced_timeout()
            (streamlined_mafia_agent.py:367-429)
            """

            # EXACT system prompts from streamlined_mafia_agent.py:372-409
            if phase == "action":
                system_prompt = f"""You are Player {getattr(self.base_agent, 'my_player_id', 'X')}, role: {getattr(self.base_agent, 'my_role', 'Unknown')}.

ACTION PHASE - Respond with [NUMBER] only.

DO NOT target yourself (Player {getattr(self.base_agent, 'my_player_id', 'X')}).

STRATEGY:
- Villager: Eliminate suspected Mafia
- Mafia: Eliminate Village power roles (Doctor/Detective)
- Doctor: Protect Village roles (not yourself unless desperate)
- Detective: Investigate suspicious players

Response format: [3] (number in brackets only, no explanation)"""
            else:
                system_prompt = f"""You are Player {getattr(self.base_agent, 'my_player_id', 'X')}, role: {getattr(self.base_agent, 'my_role', 'Unknown')}.

DISCUSSION PHASE - Natural conversation with other players.

IDENTITY:
- Speak as Player {getattr(self.base_agent, 'my_player_id', 'X')}
- Use "I" for yourself
- Address others by player number (Player 0, Player 1, etc.)

STYLE:
- Natural conversation (20-100 words)
- Make accusations, defend yourself, ask questions
- Build cases against suspected Mafia

Examples:
- "Player 2, why did you vote that way?"
- "I don't trust your explanation."
- "That doesn't make sense to me."

Goal: Win for your team through discussion and deduction."""

            # Call local LLM with exact same system prompt
            result = self.llm_engine.generate(
                observation=user_prompt,
                system_prompt=system_prompt,
                phase=phase,
            )

            if result.get('error'):
                raise Exception(f"Local LLM error: {result['error']}")

            response = result['response']
            if not response:
                raise Exception("Empty response from local LLM")

            return response

        self.base_agent._call_modal_with_enhanced_timeout = local_inference_replacement

        logger.info("✅ LocalStreamlinedMafiaAgent initialized with full Modal capabilities")

    def __call__(self, observation: str) -> str:
        """Process observation using local LLM with full Modal features"""
        return self.base_agent(observation)

    def __getattr__(self, name):
        """Delegate all other attributes to base agent"""
        return getattr(self.base_agent, name)
