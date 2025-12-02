"""
Unit tests for LLM integration components.
Tests LLM client, prompt engineering, self-bias checking, and API integration.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'bias-engine', 'src'))

from bias_engine.llm.client import LLMClient
from bias_engine.llm.models import LLMRequest, LLMResponse, ModelConfig
from bias_engine.llm.prompts import PromptEngine, PromptTemplate
from bias_engine.llm.self_bias import SelfBiasChecker
from bias_engine.llm.exceptions import LLMError, RateLimitError, ModelUnavailableError


class TestLLMClient:
    """Test suite for LLM client functionality."""

    @pytest.fixture
    def llm_client(self):
        """Create LLM client instance with mocked dependencies."""
        config = ModelConfig(
            model_name="claude-3-sonnet",
            api_endpoint="https://api.anthropic.com/v1/messages",
            max_tokens=4000,
            temperature=0.1,
            timeout=30.0
        )
        return LLMClient(config)

    @pytest.fixture
    def mock_llm_responses(self) -> Dict[str, str]:
        """Mock LLM responses for testing."""
        return {
            "bias_analysis": """
            {
                "bias_detected": true,
                "bias_types": ["confirmation_bias", "availability_bias"],
                "confidence": 0.85,
                "explanation": "The text shows strong confirmation bias by only considering evidence that supports existing beliefs.",
                "severity": "high"
            }
            """,
            "neutralized_text": """
            The research findings suggest multiple perspectives should be considered.
            While some evidence supports the hypothesis, additional studies are needed
            to reach comprehensive conclusions. Alternative viewpoints merit evaluation.
            """,
            "self_bias_check": """
            I notice I should include this self-bias check prefix: I should be mindful of potential biases in my analysis and strive for objectivity.

            [Analysis follows...]
            """,
            "error_response": "Rate limit exceeded. Please try again later."
        }

    @pytest.mark.unit
    async def test_llm_client_initialization(self, llm_client):
        """Test LLM client initialization and configuration."""
        assert llm_client.config.model_name == "claude-3-sonnet"
        assert llm_client.config.max_tokens == 4000
        assert llm_client.config.temperature == 0.1
        assert not llm_client.is_connected

    @pytest.mark.unit
    async def test_successful_llm_request(self, llm_client, mock_llm_responses):
        """Test successful LLM API request."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "content": [{"text": mock_llm_responses["bias_analysis"]}],
                "usage": {"input_tokens": 100, "output_tokens": 150}
            }

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            request = LLMRequest(
                prompt="Analyze this text for bias: 'The evidence clearly supports our conclusion.'",
                system_prompt="You are a bias detection expert.",
                temperature=0.1,
                max_tokens=1000
            )

            response = await llm_client.generate(request)

            assert isinstance(response, LLMResponse)
            assert response.success
            assert "bias_detected" in response.content
            assert response.token_usage.input_tokens == 100
            assert response.token_usage.output_tokens == 150

    @pytest.mark.unit
    async def test_llm_request_with_rate_limiting(self, llm_client, mock_llm_responses):
        """Test handling of rate limiting errors."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock rate limit response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = mock_llm_responses["error_response"]

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            request = LLMRequest(prompt="Test prompt")

            with pytest.raises(RateLimitError):
                await llm_client.generate(request)

    @pytest.mark.unit
    async def test_llm_request_timeout(self, llm_client):
        """Test handling of request timeouts."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock timeout
            mock_client.return_value.__aenter__.return_value.post.side_effect = asyncio.TimeoutError("Request timeout")

            request = LLMRequest(prompt="Test prompt", timeout=1.0)

            with pytest.raises(LLMError, match="timeout"):
                await llm_client.generate(request)

    @pytest.mark.unit
    async def test_llm_request_with_retries(self, llm_client, mock_llm_responses):
        """Test retry mechanism for failed requests."""
        with patch('httpx.AsyncClient') as mock_client:
            # First call fails, second succeeds
            mock_client_instance = mock_client.return_value.__aenter__.return_value

            # First call: server error
            fail_response = Mock()
            fail_response.status_code = 500
            fail_response.text = "Internal server error"

            # Second call: success
            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = {
                "content": [{"text": mock_llm_responses["bias_analysis"]}],
                "usage": {"input_tokens": 100, "output_tokens": 150}
            }

            mock_client_instance.post.side_effect = [fail_response, success_response]

            request = LLMRequest(prompt="Test prompt", max_retries=2)
            response = await llm_client.generate(request)

            assert response.success
            assert mock_client_instance.post.call_count == 2

    @pytest.mark.unit
    async def test_concurrent_llm_requests(self, llm_client, mock_llm_responses):
        """Test handling of concurrent LLM requests."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "content": [{"text": mock_llm_responses["bias_analysis"]}],
                "usage": {"input_tokens": 100, "output_tokens": 150}
            }

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Create multiple concurrent requests
            requests = [
                LLMRequest(prompt=f"Analyze text {i} for bias")
                for i in range(5)
            ]

            tasks = [llm_client.generate(request) for request in requests]
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            assert all(response.success for response in responses)

    @pytest.mark.unit
    async def test_token_usage_tracking(self, llm_client, mock_llm_responses):
        """Test tracking of token usage across requests."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "content": [{"text": mock_llm_responses["bias_analysis"]}],
                "usage": {"input_tokens": 150, "output_tokens": 200}
            }

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            request = LLMRequest(prompt="Test prompt")
            response = await llm_client.generate(request)

            assert response.token_usage.input_tokens == 150
            assert response.token_usage.output_tokens == 200
            assert response.token_usage.total_tokens == 350

            # Check client-level tracking
            assert llm_client.total_token_usage.total_tokens >= 350


class TestPromptEngine:
    """Test suite for prompt engineering functionality."""

    @pytest.fixture
    def prompt_engine(self):
        """Create prompt engine instance."""
        return PromptEngine()

    @pytest.fixture
    def prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Sample prompt templates for testing."""
        return {
            "bias_analysis": PromptTemplate(
                name="bias_analysis",
                template="""Analyze the following text for cognitive biases:

Text: {text}
Cultural Context: {cultural_context}

Provide analysis in JSON format with:
- bias_detected (boolean)
- bias_types (list)
- confidence (0-1)
- explanation (string)
- severity (low/medium/high)
""",
                variables=["text", "cultural_context"],
                version="1.0"
            ),
            "text_neutralization": PromptTemplate(
                name="text_neutralization",
                template="""Rewrite the following text to remove cognitive biases while preserving meaning:

Original text: {text}
Detected biases: {bias_types}
Cultural context: {cultural_context}

Provide a neutralized version that:
- Removes biased language
- Maintains original intent
- Uses inclusive language
- Acknowledges uncertainty where appropriate
""",
                variables=["text", "bias_types", "cultural_context"],
                version="1.0"
            )
        }

    @pytest.mark.unit
    async def test_prompt_template_loading(self, prompt_engine, prompt_templates):
        """Test loading and validation of prompt templates."""
        # Mock template loading
        with patch.object(prompt_engine, '_load_templates') as mock_load:
            mock_load.return_value = prompt_templates

            await prompt_engine.initialize()

            assert prompt_engine.is_initialized
            assert len(prompt_engine.templates) == 2
            assert "bias_analysis" in prompt_engine.templates
            assert "text_neutralization" in prompt_engine.templates

    @pytest.mark.unit
    async def test_prompt_template_rendering(self, prompt_engine, prompt_templates):
        """Test rendering prompts with variables."""
        prompt_engine.templates = prompt_templates
        prompt_engine.is_initialized = True

        rendered = await prompt_engine.render_prompt(
            template_name="bias_analysis",
            variables={
                "text": "This clearly proves our point.",
                "cultural_context": "en-US"
            }
        )

        assert "This clearly proves our point." in rendered
        assert "en-US" in rendered
        assert "Analyze the following text" in rendered
        assert "{text}" not in rendered  # Variables should be replaced
        assert "{cultural_context}" not in rendered

    @pytest.mark.unit
    async def test_prompt_variable_validation(self, prompt_engine, prompt_templates):
        """Test validation of prompt template variables."""
        prompt_engine.templates = prompt_templates
        prompt_engine.is_initialized = True

        # Test with missing variables
        with pytest.raises(ValueError, match="Missing required variables"):
            await prompt_engine.render_prompt(
                template_name="bias_analysis",
                variables={"text": "Test text"}  # Missing cultural_context
            )

        # Test with extra variables (should not error)
        rendered = await prompt_engine.render_prompt(
            template_name="bias_analysis",
            variables={
                "text": "Test text",
                "cultural_context": "en-US",
                "extra_var": "should be ignored"
            }
        )

        assert rendered is not None

    @pytest.mark.unit
    async def test_system_prompt_generation(self, prompt_engine):
        """Test generation of system prompts for different contexts."""
        prompt_engine.is_initialized = True

        # Test bias analysis system prompt
        bias_system = await prompt_engine.get_system_prompt("bias_analysis")
        assert "bias detection expert" in bias_system.lower()
        assert "objective" in bias_system.lower()

        # Test neutralization system prompt
        neutral_system = await prompt_engine.get_system_prompt("text_neutralization")
        assert "rewrite" in neutral_system.lower() or "neutralize" in neutral_system.lower()

    @pytest.mark.unit
    async def test_cultural_prompt_adaptation(self, prompt_engine, prompt_templates):
        """Test adaptation of prompts for different cultural contexts."""
        prompt_engine.templates = prompt_templates
        prompt_engine.is_initialized = True

        # Test with different cultural contexts
        cultures = ["en-US", "ja-JP", "de-DE"]

        for culture in cultures:
            rendered = await prompt_engine.render_prompt(
                template_name="bias_analysis",
                variables={
                    "text": "Test text",
                    "cultural_context": culture
                },
                cultural_adaptation=True
            )

            assert culture in rendered
            # Should include cultural considerations
            if culture == "ja-JP":
                assert any(word in rendered.lower() for word in ["indirect", "harmony", "consensus"])
            elif culture == "en-US":
                assert any(word in rendered.lower() for word in ["direct", "individual", "explicit"])

    @pytest.mark.unit
    async def test_prompt_template_versioning(self, prompt_engine, prompt_templates):
        """Test prompt template versioning system."""
        # Add versioned templates
        v1_template = prompt_templates["bias_analysis"]
        v2_template = PromptTemplate(
            name="bias_analysis",
            template="Enhanced analysis prompt for {text}",
            variables=["text"],
            version="2.0"
        )

        prompt_engine.templates = {
            "bias_analysis_v1.0": v1_template,
            "bias_analysis_v2.0": v2_template,
            "bias_analysis": v2_template  # Latest version
        }
        prompt_engine.is_initialized = True

        # Test getting latest version
        latest = await prompt_engine.render_prompt("bias_analysis", {"text": "test"})
        assert "Enhanced analysis" in latest

        # Test getting specific version
        v1 = await prompt_engine.render_prompt("bias_analysis_v1.0", {
            "text": "test",
            "cultural_context": "en-US"
        })
        assert "Analyze the following text" in v1


class TestSelfBiasChecker:
    """Test suite for self-bias checking functionality."""

    @pytest.fixture
    def bias_checker(self):
        """Create self-bias checker instance."""
        return SelfBiasChecker()

    @pytest.fixture
    def llm_responses_with_bias_check(self) -> Dict[str, str]:
        """LLM responses with self-bias checking."""
        return {
            "with_prefix": """I should be mindful of potential biases in my analysis and strive for objectivity.

The text shows several concerning patterns that suggest confirmation bias. The author appears to selectively present evidence that supports their predetermined conclusion while dismissing contradictory information.""",

            "without_prefix": """The text shows several concerning patterns that suggest confirmation bias. The author appears to selectively present evidence that supports their predetermined conclusion while dismissing contradictory information.""",

            "incorrect_prefix": """I need to be careful about my own biases here.

The analysis reveals multiple bias indicators...""",

            "empty_response": "",

            "valid_with_explanation": """I should be mindful of potential biases in my analysis and strive for objectivity.

Based on systematic analysis of linguistic patterns and semantic structures, the text exhibits characteristics of confirmation bias (confidence: 0.85) and availability bias (confidence: 0.72). The author uses definitive language like 'clearly' and 'obviously' when presenting debatable claims."""
        }

    @pytest.mark.unit
    async def test_self_bias_prefix_validation(self, bias_checker, llm_responses_with_bias_check):
        """Test validation of self-bias check prefix."""
        # Test valid prefix
        valid_response = llm_responses_with_bias_check["with_prefix"]
        is_valid, explanation = await bias_checker.validate_self_bias_check(valid_response)
        assert is_valid
        assert explanation is None

        # Test missing prefix
        invalid_response = llm_responses_with_bias_check["without_prefix"]
        is_valid, explanation = await bias_checker.validate_self_bias_check(invalid_response)
        assert not is_valid
        assert "missing self-bias check prefix" in explanation.lower()

        # Test incorrect prefix
        incorrect_response = llm_responses_with_bias_check["incorrect_prefix"]
        is_valid, explanation = await bias_checker.validate_self_bias_check(incorrect_response)
        assert not is_valid
        assert "incorrect prefix format" in explanation.lower()

    @pytest.mark.unit
    async def test_self_bias_prefix_injection(self, bias_checker):
        """Test automatic injection of self-bias check prefix."""
        original_prompt = "Analyze this text for cognitive biases: 'The evidence clearly supports our conclusion.'"

        enhanced_prompt = await bias_checker.inject_self_bias_prompt(original_prompt)

        assert "self-bias check prefix" in enhanced_prompt.lower()
        assert "I should be mindful of potential biases" in enhanced_prompt
        assert original_prompt in enhanced_prompt

    @pytest.mark.unit
    async def test_self_bias_compliance_enforcement(self, bias_checker, llm_responses_with_bias_check):
        """Test enforcement of self-bias compliance in LLM responses."""
        # Test compliance with valid response
        valid_response = llm_responses_with_bias_check["valid_with_explanation"]

        compliant_response = await bias_checker.enforce_compliance(valid_response)
        assert compliant_response == valid_response  # No changes needed

        # Test non-compliant response gets fixed
        non_compliant = llm_responses_with_bias_check["without_prefix"]

        fixed_response = await bias_checker.enforce_compliance(non_compliant)
        assert "I should be mindful of potential biases" in fixed_response
        assert non_compliant in fixed_response

    @pytest.mark.unit
    async def test_self_bias_check_quality_assessment(self, bias_checker, llm_responses_with_bias_check):
        """Test quality assessment of self-bias checks."""
        responses_to_test = [
            llm_responses_with_bias_check["with_prefix"],
            llm_responses_with_bias_check["valid_with_explanation"],
            llm_responses_with_bias_check["incorrect_prefix"]
        ]

        for response in responses_to_test:
            quality_score = await bias_checker.assess_self_bias_quality(response)
            assert 0.0 <= quality_score <= 1.0

            if "I should be mindful of potential biases" in response:
                assert quality_score >= 0.8
            else:
                assert quality_score <= 0.3

    @pytest.mark.unit
    async def test_self_bias_metadata_extraction(self, bias_checker, llm_responses_with_bias_check):
        """Test extraction of self-bias metadata from responses."""
        response = llm_responses_with_bias_check["valid_with_explanation"]

        metadata = await bias_checker.extract_self_bias_metadata(response)

        assert "has_prefix" in metadata
        assert metadata["has_prefix"] is True
        assert "prefix_quality" in metadata
        assert metadata["prefix_quality"] >= 0.8
        assert "compliance_score" in metadata
        assert 0.0 <= metadata["compliance_score"] <= 1.0

    @pytest.mark.unit
    async def test_batch_self_bias_validation(self, bias_checker, llm_responses_with_bias_check):
        """Test batch validation of multiple responses."""
        responses = list(llm_responses_with_bias_check.values())

        validation_results = await bias_checker.batch_validate(responses)

        assert len(validation_results) == len(responses)

        for i, (is_valid, explanation) in enumerate(validation_results):
            response = responses[i]
            if "I should be mindful of potential biases" in response:
                assert is_valid
            else:
                assert not is_valid
                assert explanation is not None


# Integration tests
@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests for complete LLM workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_bias_analysis(self):
        """Test complete bias analysis workflow with LLM."""
        # Mock LLM client
        mock_client = Mock(spec=LLMClient)
        mock_response = LLMResponse(
            content="""I should be mindful of potential biases in my analysis and strive for objectivity.

{
    "bias_detected": true,
    "bias_types": ["confirmation_bias"],
    "confidence": 0.85,
    "explanation": "Text shows confirmation bias through selective evidence presentation",
    "severity": "high"
}""",
            success=True,
            token_usage={"input_tokens": 150, "output_tokens": 200}
        )
        mock_client.generate = AsyncMock(return_value=mock_response)

        # Mock prompt engine
        mock_prompt_engine = Mock(spec=PromptEngine)
        mock_prompt_engine.render_prompt = AsyncMock(return_value="Test prompt")
        mock_prompt_engine.get_system_prompt = AsyncMock(return_value="System prompt")

        # Mock self-bias checker
        mock_bias_checker = Mock(spec=SelfBiasChecker)
        mock_bias_checker.inject_self_bias_prompt = AsyncMock(return_value="Enhanced prompt")
        mock_bias_checker.validate_self_bias_check = AsyncMock(return_value=(True, None))

        # Test workflow
        from bias_engine.llm.pipeline import LLMPipeline
        pipeline = LLMPipeline(mock_client, mock_prompt_engine, mock_bias_checker)

        result = await pipeline.analyze_bias(
            text="This evidence clearly proves our theory is correct.",
            cultural_context="en-US"
        )

        # Verify workflow completion
        assert result is not None
        assert result.get("bias_detected") is True
        assert "confirmation_bias" in result.get("bias_types", [])

        # Verify all components were called
        mock_prompt_engine.render_prompt.assert_called_once()
        mock_bias_checker.inject_self_bias_prompt.assert_called_once()
        mock_client.generate.assert_called_once()
        mock_bias_checker.validate_self_bias_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_error_handling_workflow(self):
        """Test error handling in complete LLM workflow."""
        # Mock LLM client that fails
        mock_client = Mock(spec=LLMClient)
        mock_client.generate = AsyncMock(side_effect=RateLimitError("Rate limit exceeded"))

        mock_prompt_engine = Mock(spec=PromptEngine)
        mock_prompt_engine.render_prompt = AsyncMock(return_value="Test prompt")

        mock_bias_checker = Mock(spec=SelfBiasChecker)
        mock_bias_checker.inject_self_bias_prompt = AsyncMock(return_value="Enhanced prompt")

        from bias_engine.llm.pipeline import LLMPipeline
        pipeline = LLMPipeline(mock_client, mock_prompt_engine, mock_bias_checker)

        # Should handle error gracefully
        with pytest.raises(RateLimitError):
            await pipeline.analyze_bias("Test text", "en-US")


# Performance tests
@pytest.mark.slow
class TestLLMPerformance:
    """Performance tests for LLM integration."""

    @pytest.mark.asyncio
    async def test_llm_response_latency(self):
        """Test LLM response latency under normal conditions."""
        # Mock fast LLM responses
        mock_client = Mock(spec=LLMClient)

        import time

        async def mock_generate(request):
            await asyncio.sleep(0.1)  # Simulate network delay
            return LLMResponse(
                content="Mock response",
                success=True,
                token_usage={"input_tokens": 100, "output_tokens": 50}
            )

        mock_client.generate = mock_generate

        # Test multiple requests
        requests = [LLMRequest(prompt=f"Test prompt {i}") for i in range(10)]

        start_time = time.time()
        tasks = [mock_client.generate(request) for request in requests]
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time = end_time - start_time
        avg_latency = total_time / len(responses)

        print(f"Average LLM latency: {avg_latency:.3f}s")

        # Performance targets
        assert avg_latency < 5.0  # Average latency should be under 5 seconds
        assert all(response.success for response in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])