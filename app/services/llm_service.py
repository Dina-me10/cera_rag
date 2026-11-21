import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover - adapter ishlatilmasa peft shart emas
    PeftConfig = None
    PeftModel = None

from app.core.config import Settings


class LocalLLM:
    """TinyLlama + CeraGPT adapter (LoRA daftari) bilan javob yozadigan model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.device = self._resolve_device(settings.llm_device)
        self._dtype = torch.float16 if self.device in ('mps', 'cuda') else torch.float32
        model, tokenizer = self._load_model_and_tokenizer(settings)
        model = model.to(self.device)
        model.eval()
        self._pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=self.device,
        )

    @staticmethod
    def _resolve_device(requested: str | None) -> str:
        requested = (requested or 'auto').lower()
        if requested in ('auto', 'mps') and torch.backends.mps.is_available():
            return 'mps'
        if requested in ('auto', 'cuda') and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def generate(self, prompt: str) -> str:
        generator = self._pipeline
        outputs = generator(
            prompt,
            max_new_tokens=self._settings.llm_max_new_tokens,
            temperature=self._settings.llm_temperature,
            do_sample=self._settings.llm_temperature > 0,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        text = outputs[0]['generated_text']
        return text[len(prompt):].strip()

    def _load_model_and_tokenizer(self, settings: Settings):
        adapter_path = settings.llm_adapter_path
        # LLM_ADAPTER_PATH bo'sh bo'lsa ('' yoki None) -> oddiy TinyLlama
        if not adapter_path:
            tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
            model = AutoModelForCausalLM.from_pretrained(settings.llm_model, dtype=self._dtype)
            return model, tokenizer

        if PeftModel is None or PeftConfig is None:
            raise RuntimeError(
                'CeraGPT adapter requested but "peft" is not installed. '
                'Install dependencies from requirements.txt.'
            )

        base_model_name = settings.llm_model
        try:
            peft_config = PeftConfig.from_pretrained(adapter_path)
            if peft_config.base_model_name_or_path:
                base_model_name = peft_config.base_model_name_or_path
        except Exception:
            peft_config = None

        # 1) asl TinyLlama (miya)  2) unga CeraGPT daftarini ulaymiz
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=self._dtype)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        return model, tokenizer
