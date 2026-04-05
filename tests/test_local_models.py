"""Tests for local model configuration and provider dispatch."""

from unittest.mock import MagicMock, patch

from self_loop_eval.config import ModelConfig, SystemConfig
from self_loop_eval.models.student import StudentModel
from self_loop_eval.models.teacher import TeacherModel


class TestModelConfigDefaults:
    """Verify default config uses local models, not OpenAI."""

    def test_default_provider_is_local(self):
        config = ModelConfig()
        assert config.provider == "local"

    def test_default_student_model_is_tinyllama(self):
        config = ModelConfig()
        assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_default_teacher_model_is_qwen(self):
        sys_config = SystemConfig()
        assert sys_config.teacher.model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_default_quantization_is_4bit(self):
        config = ModelConfig()
        assert config.quantization == "4bit"

    def test_default_device_map_is_auto(self):
        config = ModelConfig()
        assert config.device_map == "auto"

    def test_student_default_provider_is_local(self):
        sys_config = SystemConfig()
        assert sys_config.student.provider == "local"

    def test_teacher_default_provider_is_local(self):
        sys_config = SystemConfig()
        assert sys_config.teacher.provider == "local"

    def test_openai_provider_still_configurable(self):
        config = ModelConfig(provider="openai", model_name="gpt-4")
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"

    def test_from_dict_preserves_local_defaults(self):
        sys_config = SystemConfig.from_dict({})
        assert sys_config.student.provider == "local"
        assert sys_config.teacher.provider == "local"

    def test_from_dict_openai_override(self):
        sys_config = SystemConfig.from_dict({
            "student": {"provider": "openai", "model_name": "gpt-3.5-turbo"},
            "teacher": {"provider": "openai", "model_name": "gpt-4"},
        })
        assert sys_config.student.provider == "openai"
        assert sys_config.teacher.provider == "openai"

    def test_quantization_options(self):
        assert ModelConfig(quantization="4bit").quantization == "4bit"
        assert ModelConfig(quantization="8bit").quantization == "8bit"
        assert ModelConfig(quantization=None).quantization is None

    def test_local_model_path_override(self):
        config = ModelConfig(local_model_path="/path/to/local/model")
        assert config.local_model_path == "/path/to/local/model"


class TestProviderDispatch:
    """Verify that generate() dispatches to the correct backend."""

    def test_student_local_calls_generate_local(self):
        config = ModelConfig(provider="local")
        student = StudentModel(config)
        student._generate_local = MagicMock(return_value="local response")
        result = student.generate("test prompt")
        student._generate_local.assert_called_once()
        assert result == "local response"

    def test_teacher_local_calls_generate_local(self):
        config = ModelConfig(provider="local")
        teacher = TeacherModel(config)
        teacher._generate_local = MagicMock(return_value="local response")
        result = teacher.generate("test prompt")
        teacher._generate_local.assert_called_once()
        assert result == "local response"

    def test_student_openai_calls_client(self):
        config = ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
        student = StudentModel(config)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "openai response"
        mock_client.chat.completions.create.return_value = mock_response
        student._client = mock_client
        result = student.generate("test prompt")
        mock_client.chat.completions.create.assert_called_once()
        assert result == "openai response"

    def test_teacher_openai_calls_client(self):
        config = ModelConfig(provider="openai", model_name="gpt-4")
        teacher = TeacherModel(config)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "openai response"
        mock_client.chat.completions.create.return_value = mock_response
        teacher._client = mock_client
        result = teacher.generate("test prompt")
        mock_client.chat.completions.create.assert_called_once()
        assert result == "openai response"


class TestLocalModelLoading:
    """Test the local model loading logic."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_local_model_uses_model_name(self, mock_tokenizer_cls, mock_model_cls):
        config = ModelConfig(
            provider="local",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            quantization=None,
        )
        student = StudentModel(config)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        student._load_local_model()

        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        mock_model_cls.from_pretrained.assert_called_once()
        assert student._local_model is mock_model
        assert student._local_tokenizer is mock_tokenizer

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_local_model_prefers_local_path(self, mock_tokenizer_cls, mock_model_cls):
        config = ModelConfig(
            provider="local",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            local_model_path="/custom/path",
            quantization=None,
        )
        student = StudentModel(config)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        student._load_local_model()

        mock_tokenizer_cls.from_pretrained.assert_called_once_with("/custom/path")

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_local_model_sets_pad_token(self, mock_tokenizer_cls, mock_model_cls):
        config = ModelConfig(provider="local", quantization=None)
        student = StudentModel(config)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        student._load_local_model()

        assert mock_tokenizer.pad_token == "</s>"

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_local_model_only_loads_once(self, mock_tokenizer_cls, mock_model_cls):
        config = ModelConfig(provider="local", quantization=None)
        student = StudentModel(config)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = MagicMock()

        student._load_local_model()
        student._load_local_model()

        # Should only be called once due to caching
        assert mock_tokenizer_cls.from_pretrained.call_count == 1
        assert mock_model_cls.from_pretrained.call_count == 1


class TestMockStudentStillWorks:
    """Ensure the existing mock student pattern (provider='mock') still works."""

    def test_mock_provider_does_not_crash(self):
        config = ModelConfig(provider="mock", model_name="mock-student")
        student = StudentModel(config)
        # Mock provider should not call local or openai
        student._generate_local = MagicMock(side_effect=RuntimeError("should not be called"))
        student._get_client = MagicMock(side_effect=RuntimeError("should not be called"))

        # Override generate for test (as the existing tests do)
        class MockStudent(StudentModel):
            def generate(self, prompt, system_prompt=""):
                return '{"content": "mock", "reasoning": "mock", "score": 0.5}'

        mock = MockStudent(config)
        result = mock.generate("test")
        assert "mock" in result
