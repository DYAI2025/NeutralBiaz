# LLM Prompting Implementation Guide

## Overview

This guide provides concrete implementation strategies for deploying the enhanced prompting framework in the BiasNeutralize AI system. It covers integration patterns, testing methodologies, and operational best practices.

## 1. Integration Architecture

### 1.1 Prompt Management System

```python
# Example implementation structure
class PromptManager:
    def __init__(self, config_path: str):
        self.prompts = self._load_prompts(config_path)
        self.validators = self._setup_validators()
        self.safety_checkers = self._setup_safety_checkers()

    def get_prompt(self, prompt_type: str, context: dict) -> str:
        """Get properly formatted prompt with variable substitution"""
        template = self.prompts[prompt_type]
        validated_context = self.validators[prompt_type].validate(context)
        return self._substitute_variables(template, validated_context)

    def validate_output(self, output: str, prompt_type: str) -> ValidationResult:
        """Validate LLM output against safety and quality criteria"""
        return self.safety_checkers[prompt_type].check(output)
```

### 1.2 Prompt Chain Orchestration

```python
class BiasNeutralizationPipeline:
    def __init__(self, llm_client, prompt_manager):
        self.llm = llm_client
        self.prompts = prompt_manager

    async def process_span(self, span_data: dict) -> dict:
        """Process a single bias span through the full pipeline"""

        # Stage 1: Input validation
        validation_result = self.prompts.validate_input(span_data)
        if not validation_result.is_valid:
            return self._handle_validation_error(validation_result)

        # Stage 2: Cultural context enrichment
        cultural_context = await self._enrich_cultural_context(span_data)

        # Stage 3: Bias analysis and neutralization
        analysis_prompt = self.prompts.get_prompt("debias_span_enhanced", {
            **span_data,
            **cultural_context
        })

        analysis_result = await self.llm.complete(
            messages=[
                {"role": "system", "content": self.prompts.get_system_prompt()},
                {"role": "user", "content": analysis_prompt}
            ]
        )

        # Stage 4: Self-bias check
        bias_check_result = await self._perform_self_bias_check(
            analysis_result, span_data
        )

        # Stage 5: Final validation
        final_validation = self.prompts.validate_output(
            bias_check_result, "debias_span_enhanced"
        )

        return self._format_final_result(bias_check_result, final_validation)
```

## 2. Variable Management and Validation

### 2.1 Context Variable Schema

```python
from pydantic import BaseModel, validator
from typing import Optional, List, Union

class BiasAnalysisContext(BaseModel):
    # Required fields
    input_language: str
    output_language: str
    bias_family: str
    bias_subtype: str
    bias_span: str
    full_sentence_or_paragraph: str

    # Cultural context
    sender_culture: str
    receiver_culture: str
    context_topic: Optional[str]
    audience: Optional[str]
    formality_level: Optional[str] = "neutral"

    # Bias metadata
    severity_raw: float
    severity_sender: Optional[float]
    severity_receiver: Optional[float]
    cultural_explanation: Optional[str]

    # System metadata
    span_id: str

    @validator('input_language', 'output_language')
    def validate_language_code(cls, v):
        if not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError('Invalid language code format')
        return v

    @validator('severity_raw', 'severity_sender', 'severity_receiver')
    def validate_severity(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Severity must be between 0 and 10')
        return v

    @validator('bias_family')
    def validate_bias_family(cls, v):
        valid_families = [
            "Racism", "Sexism", "Classism", "Ableism", "Ageism",
            "Queerfeindlichkeit", "Xenophobie", "Religious", "Other"
        ]
        if v not in valid_families:
            raise ValueError(f'Invalid bias family: {v}')
        return v
```

### 2.2 Dynamic Variable Substitution

```python
class PromptVariableSubstitutor:
    def __init__(self):
        self.fallback_values = {
            'formality_level': 'neutral',
            'audience': 'general public',
            'context_topic': 'general discussion'
        }

    def substitute(self, template: str, context: dict) -> str:
        """Safely substitute variables with fallbacks and validation"""

        # Extract required variables from template
        required_vars = re.findall(r'\{\{(\w+)\}\}', template)

        # Prepare substitution context with fallbacks
        subst_context = self._prepare_context(context, required_vars)

        # Perform substitution with safety checks
        try:
            return template.format(**subst_context)
        except KeyError as e:
            raise PromptVariableError(f"Missing required variable: {e}")

    def _prepare_context(self, context: dict, required_vars: List[str]) -> dict:
        """Prepare context with fallbacks and validation"""
        result = {}

        for var in required_vars:
            if var in context:
                result[var] = self._sanitize_value(context[var])
            elif var in self.fallback_values:
                result[var] = self.fallback_values[var]
                logging.warning(f"Using fallback for variable: {var}")
            else:
                raise PromptVariableError(f"Required variable missing: {var}")

        return result

    def _sanitize_value(self, value: any) -> str:
        """Sanitize variable values for prompt injection safety"""
        if isinstance(value, str):
            # Remove potential prompt injection patterns
            sanitized = re.sub(r'[{}]', '', value)  # Remove braces
            sanitized = re.sub(r'[\r\n]+', ' ', sanitized)  # Normalize whitespace
            return sanitized[:500]  # Limit length
        return str(value)
```

## 3. Cultural Context Management

### 3.1 Cultural Profile Integration

```python
class CulturalContextManager:
    def __init__(self, profiles_path: str):
        self.profiles = self._load_cultural_profiles(profiles_path)
        self.risk_assessor = CulturalRiskAssessor()

    async def enrich_context(self, sender: str, receiver: str, content: dict) -> dict:
        """Enrich context with cultural factors and risk assessment"""

        sender_profile = self.profiles.get(sender, self._get_default_profile())
        receiver_profile = self.profiles.get(receiver, self._get_default_profile())

        cultural_distance = self._calculate_cultural_distance(
            sender_profile, receiver_profile
        )

        risk_factors = self.risk_assessor.assess(
            sender_profile, receiver_profile, content
        )

        return {
            'cultural_distance': cultural_distance,
            'hofstede_sender': sender_profile.hofstede,
            'hofstede_receiver': receiver_profile.hofstede,
            'risk_factors': risk_factors,
            'cultural_bridge_needed': cultural_distance > 0.3
        }

    def _calculate_cultural_distance(self, profile1, profile2) -> float:
        """Calculate cultural distance based on Hofstede dimensions"""
        dimensions = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']

        total_distance = 0
        for dim in dimensions:
            dist = abs(profile1.hofstede[dim] - profile2.hofstede[dim]) / 100
            total_distance += dist ** 2

        return (total_distance / len(dimensions)) ** 0.5
```

### 3.2 Cultural Risk Assessment

```python
class CulturalRiskAssessor:
    def __init__(self):
        self.risk_patterns = self._load_risk_patterns()

    def assess(self, sender_profile, receiver_profile, content: dict) -> List[str]:
        """Assess cultural risks for cross-cultural communication"""

        risks = []

        # Directness risks
        if self._is_direct_culture(sender_profile) and self._is_indirect_culture(receiver_profile):
            if content.get('emotional_intensity', 0) > 0.6:
                risks.append('directness_mismatch')

        # Hierarchy risks
        hierarchy_diff = abs(
            sender_profile.hofstede['pdi'] - receiver_profile.hofstede['pdi']
        )
        if hierarchy_diff > 30 and 'authority' in content.get('topic_tags', []):
            risks.append('hierarchy_sensitivity')

        # Individual/collective risks
        idv_diff = abs(
            sender_profile.hofstede['idv'] - receiver_profile.hofstede['idv']
        )
        if idv_diff > 40 and 'group_reference' in content.get('content_features', []):
            risks.append('individualism_collectivism_clash')

        return risks

    def _is_direct_culture(self, profile) -> bool:
        """Determine if culture tends toward direct communication"""
        return profile.country_code in ['DE', 'NL', 'DK', 'NO']

    def _is_indirect_culture(self, profile) -> bool:
        """Determine if culture tends toward indirect communication"""
        return profile.country_code in ['JP', 'TH', 'KR', 'CN']
```

## 4. Safety and Quality Validation

### 4.1 Multi-Layer Validation System

```python
class ComprehensiveValidator:
    def __init__(self):
        self.safety_validator = SafetyValidator()
        self.quality_validator = QualityValidator()
        self.cultural_validator = CulturalValidator()
        self.epistemic_validator = EpistemicValidator()

    async def validate_full(self, output: dict, context: dict) -> ValidationResult:
        """Perform comprehensive validation of LLM output"""

        results = await asyncio.gather(
            self.safety_validator.validate(output),
            self.quality_validator.validate(output, context),
            self.cultural_validator.validate(output, context),
            self.epistemic_validator.validate(output)
        )

        return ValidationResult.combine(results)

class SafetyValidator:
    def __init__(self):
        self.toxic_patterns = self._load_toxic_patterns()
        self.bias_detectors = self._setup_bias_detectors()

    async def validate(self, output: dict) -> SafetyValidationResult:
        """Validate output for safety issues"""

        issues = []

        # Check for introduced bias
        for variant in ['variant_A_neutral', 'variant_B_emotional']:
            if variant in output.get('neutralization', {}):
                text = output['neutralization'][variant]['text']
                bias_score = await self._detect_new_bias(text)
                if bias_score > 0.3:
                    issues.append(f"Potential bias introduced in {variant}")

        # Check for toxic language
        toxic_score = self._check_toxicity(output)
        if toxic_score > 0.2:
            issues.append("Toxic language detected in output")

        # Check safety score
        safety_score = output.get('quality_metrics', {}).get('safety_score', 0)
        if safety_score < 0.8:
            issues.append("Safety score below threshold")

        return SafetyValidationResult(
            is_safe=len(issues) == 0,
            safety_score=safety_score,
            issues=issues
        )
```

### 4.2 Epistemic Validation

```python
class EpistemicValidator:
    def __init__(self):
        self.fact_checker = FactChecker()
        self.confidence_calibrator = ConfidenceCalibrator()

    async def validate(self, output: dict) -> EpistemicValidationResult:
        """Validate epistemic quality of output"""

        # Extract claims from analysis and variants
        claims = self._extract_claims(output)

        # Validate epistemic markers
        marker_validation = self._validate_epistemic_markers(claims)

        # Check confidence calibration
        confidence_validation = self._validate_confidence_levels(output)

        # Check for overconfidence
        overconfidence_issues = self._check_overconfidence(claims)

        return EpistemicValidationResult(
            marker_compliance=marker_validation.compliance_rate,
            confidence_calibration=confidence_validation.calibration_score,
            overconfidence_flags=overconfidence_issues,
            overall_epistemic_quality=self._calculate_overall_score(
                marker_validation, confidence_validation, overconfidence_issues
            )
        )

    def _validate_epistemic_markers(self, claims: List[str]) -> MarkerValidation:
        """Validate that claims have appropriate epistemic markers"""

        required_prefixes = [
            "Faktisch korrekt", "Nachweislich", "Dokumentiert",  # Factual
            "Logisch scheint", "Daraus folgt", "Plausibel ist",   # Logical
            "Rein subjektiv", "Aus meiner Sicht", "Möglicherweise"  # Subjective
        ]

        marked_claims = 0
        for claim in claims:
            if any(prefix in claim for prefix in required_prefixes):
                marked_claims += 1

        compliance_rate = marked_claims / len(claims) if claims else 1.0

        return MarkerValidation(
            total_claims=len(claims),
            marked_claims=marked_claims,
            compliance_rate=compliance_rate
        )
```

## 5. Error Handling and Recovery

### 5.1 Graceful Degradation Patterns

```python
class ErrorRecoveryManager:
    def __init__(self, prompt_manager):
        self.prompts = prompt_manager
        self.fallback_strategies = {
            'input_validation_failed': self._handle_input_validation_error,
            'cultural_profile_missing': self._handle_missing_cultural_profile,
            'llm_generation_failed': self._handle_generation_failure,
            'output_validation_failed': self._handle_output_validation_error
        }

    async def handle_error(self, error_type: str, context: dict) -> dict:
        """Handle errors with appropriate fallback strategy"""

        if error_type in self.fallback_strategies:
            return await self.fallback_strategies[error_type](context)
        else:
            return self._create_safe_default_response(error_type, context)

    async def _handle_input_validation_error(self, context: dict) -> dict:
        """Handle input validation failures"""

        # Try to fix common issues
        fixed_context = self._attempt_input_repair(context)

        if fixed_context:
            logging.info("Successfully repaired input validation issues")
            return {"status": "repaired", "context": fixed_context}

        # If repair fails, return safe minimal response
        return {
            "status": "validation_failed",
            "error": "Input validation failed",
            "safe_response": {
                "span_id": context.get("span_id", "unknown"),
                "safety_notes": "Input could not be validated safely",
                "human_review_required": True
            }
        }

    async def _handle_missing_cultural_profile(self, context: dict) -> dict:
        """Handle missing cultural profiles"""

        # Use neutral baseline
        neutral_profile = self._get_neutral_cultural_baseline()

        modified_context = {
            **context,
            'sender_culture': 'neutral',
            'receiver_culture': 'neutral',
            'cultural_explanation': 'Cultural profiles not available, using neutral baseline'
        }

        return {
            "status": "degraded_mode",
            "context": modified_context,
            "limitations": ["Cultural analysis not available"]
        }
```

### 5.2 Monitoring and Alerting

```python
class PromptPerformanceMonitor:
    def __init__(self):
        self.metrics = PrometheusMetrics()
        self.alert_manager = AlertManager()

    def track_prompt_execution(self, prompt_type: str, execution_time: float,
                             success: bool, quality_score: float):
        """Track prompt execution metrics"""

        self.metrics.record_prompt_latency(prompt_type, execution_time)
        self.metrics.record_prompt_success_rate(prompt_type, success)
        self.metrics.record_quality_score(prompt_type, quality_score)

        # Check for performance degradation
        if execution_time > self._get_latency_threshold(prompt_type):
            self.alert_manager.send_alert(
                "prompt_latency_high",
                f"{prompt_type} execution time: {execution_time}s"
            )

        if quality_score < self._get_quality_threshold(prompt_type):
            self.alert_manager.send_alert(
                "prompt_quality_low",
                f"{prompt_type} quality score: {quality_score}"
            )

    def track_validation_failures(self, validation_type: str, failure_reasons: List[str]):
        """Track validation failure patterns"""

        for reason in failure_reasons:
            self.metrics.increment_validation_failure(validation_type, reason)

        # Alert on systematic validation failures
        failure_rate = self.metrics.get_validation_failure_rate(validation_type)
        if failure_rate > 0.1:  # 10% failure rate
            self.alert_manager.send_alert(
                "validation_failure_rate_high",
                f"{validation_type} failure rate: {failure_rate:.2%}"
            )
```

## 6. Testing and Evaluation Framework

### 6.1 Automated Testing Suite

```python
class PromptTestingSuite:
    def __init__(self, test_data_path: str):
        self.test_cases = self._load_test_cases(test_data_path)
        self.evaluators = self._setup_evaluators()

    async def run_full_suite(self) -> TestResults:
        """Run comprehensive testing suite"""

        results = TestResults()

        # Test individual prompt types
        for prompt_type in ['debias_span', 'debias_batch', 'marker_generator']:
            prompt_results = await self._test_prompt_type(prompt_type)
            results.add_prompt_results(prompt_type, prompt_results)

        # Test end-to-end workflows
        e2e_results = await self._test_end_to_end_workflows()
        results.add_e2e_results(e2e_results)

        # Test error handling
        error_handling_results = await self._test_error_handling()
        results.add_error_handling_results(error_handling_results)

        return results

    async def _test_prompt_type(self, prompt_type: str) -> PromptTypeResults:
        """Test specific prompt type with various inputs"""

        test_cases = self.test_cases.get_cases_for_prompt(prompt_type)
        results = PromptTypeResults(prompt_type)

        for test_case in test_cases:
            try:
                # Execute prompt
                output = await self._execute_test_case(prompt_type, test_case)

                # Evaluate output
                evaluation = self.evaluators[prompt_type].evaluate(
                    output, test_case.expected
                )

                results.add_case_result(test_case.id, evaluation)

            except Exception as e:
                results.add_error(test_case.id, str(e))

        return results
```

### 6.2 Cultural Bias Testing

```python
class CulturalBiasTestSuite:
    def __init__(self):
        self.cultural_test_scenarios = self._create_test_scenarios()
        self.bias_detectors = self._setup_bias_detectors()

    async def test_cultural_fairness(self) -> CulturalFairnessResults:
        """Test for cultural bias in prompt outputs"""

        results = CulturalFairnessResults()

        for scenario in self.cultural_test_scenarios:
            # Test same content across different cultural contexts
            outputs = []
            for culture_pair in scenario.culture_pairs:
                context = {
                    **scenario.base_context,
                    'sender_culture': culture_pair[0],
                    'receiver_culture': culture_pair[1]
                }

                output = await self._generate_output(context)
                outputs.append((culture_pair, output))

            # Analyze outputs for cultural bias
            bias_analysis = self._analyze_cultural_bias(outputs)
            results.add_scenario_result(scenario.id, bias_analysis)

        return results

    def _analyze_cultural_bias(self, outputs: List[Tuple]) -> CulturalBiasAnalysis:
        """Analyze outputs for systematic cultural bias"""

        # Check for consistent favoritism toward specific cultures
        severity_by_culture = defaultdict(list)
        for (sender, receiver), output in outputs:
            severity = output.get('bias_analysis', {}).get('severity', 0)
            severity_by_culture[f"{sender}->{receiver}"].append(severity)

        # Statistical analysis of bias patterns
        bias_patterns = {}
        for culture_pair, severities in severity_by_culture.items():
            mean_severity = np.mean(severities)
            std_severity = np.std(severities)
            bias_patterns[culture_pair] = {
                'mean_severity': mean_severity,
                'std_severity': std_severity,
                'sample_size': len(severities)
            }

        # Detect systematic bias
        systematic_bias = self._detect_systematic_bias(bias_patterns)

        return CulturalBiasAnalysis(
            bias_patterns=bias_patterns,
            systematic_bias_detected=systematic_bias,
            recommendations=self._generate_bias_mitigation_recommendations(bias_patterns)
        )
```

## 7. Performance Optimization

### 7.1 Caching Strategy

```python
class PromptCacheManager:
    def __init__(self, redis_client):
        self.cache = redis_client
        self.cache_policies = {
            'cultural_profiles': {'ttl': 86400, 'key_pattern': 'culture:{culture_code}'},
            'bias_patterns': {'ttl': 3600, 'key_pattern': 'bias:{bias_hash}'},
            'llm_responses': {'ttl': 1800, 'key_pattern': 'llm:{prompt_hash}'}
        }

    async def get_cached_result(self, cache_type: str, key_data: dict) -> Optional[dict]:
        """Retrieve cached result if available"""

        cache_key = self._build_cache_key(cache_type, key_data)
        cached_data = await self.cache.get(cache_key)

        if cached_data:
            return json.loads(cached_data)

        return None

    async def cache_result(self, cache_type: str, key_data: dict, result: dict):
        """Cache result with appropriate TTL"""

        cache_key = self._build_cache_key(cache_type, key_data)
        ttl = self.cache_policies[cache_type]['ttl']

        await self.cache.setex(
            cache_key,
            ttl,
            json.dumps(result, default=str)
        )

    def _build_cache_key(self, cache_type: str, key_data: dict) -> str:
        """Build consistent cache key"""

        pattern = self.cache_policies[cache_type]['key_pattern']

        if cache_type == 'llm_responses':
            # Hash the prompt content for consistent keys
            prompt_content = json.dumps(key_data, sort_keys=True)
            prompt_hash = hashlib.sha256(prompt_content.encode()).hexdigest()[:16]
            return pattern.format(prompt_hash=prompt_hash)

        return pattern.format(**key_data)
```

### 7.2 Batch Processing Optimization

```python
class BatchProcessor:
    def __init__(self, llm_client, max_batch_size: int = 5):
        self.llm = llm_client
        self.max_batch_size = max_batch_size

    async def process_batch(self, spans: List[dict]) -> List[dict]:
        """Process multiple spans efficiently in batches"""

        # Group spans by similar characteristics
        batches = self._create_optimal_batches(spans)

        # Process batches concurrently
        results = await asyncio.gather(*[
            self._process_single_batch(batch) for batch in batches
        ])

        # Flatten and return results
        return [result for batch_results in results for result in batch_results]

    def _create_optimal_batches(self, spans: List[dict]) -> List[List[dict]]:
        """Create optimal batches based on similarity"""

        # Group by bias family and cultural context for efficiency
        groups = defaultdict(list)
        for span in spans:
            group_key = (
                span.get('bias_family'),
                span.get('sender_culture'),
                span.get('receiver_culture')
            )
            groups[group_key].append(span)

        # Split large groups into max_batch_size chunks
        batches = []
        for group_spans in groups.values():
            for i in range(0, len(group_spans), self.max_batch_size):
                batches.append(group_spans[i:i + self.max_batch_size])

        return batches

    async def _process_single_batch(self, spans: List[dict]) -> List[dict]:
        """Process a single batch of similar spans"""

        if len(spans) == 1:
            # Use single span prompt for individual items
            return [await self._process_single_span(spans[0])]
        else:
            # Use batch prompt for multiple items
            return await self._process_batch_prompt(spans)
```

## 8. Deployment and Monitoring

### 8.1 Production Deployment Configuration

```python
class ProductionPromptConfig:
    def __init__(self, environment: str):
        self.environment = environment
        self.config = self._load_environment_config(environment)

    def get_llm_config(self) -> dict:
        """Get LLM configuration for production"""
        return {
            'model': self.config.get('llm_model', 'gpt-4'),
            'temperature': self.config.get('temperature', 0.3),
            'max_tokens': self.config.get('max_tokens', 2000),
            'timeout': self.config.get('timeout', 30),
            'retry_attempts': self.config.get('retry_attempts', 3),
            'safety_filters': self.config.get('safety_filters', True)
        }

    def get_validation_config(self) -> dict:
        """Get validation configuration for production"""
        return {
            'safety_threshold': self.config.get('safety_threshold', 0.8),
            'quality_threshold': self.config.get('quality_threshold', 0.7),
            'cultural_appropriateness_threshold': 0.7,
            'epistemic_compliance_threshold': 0.9,
            'enable_human_review_flags': True
        }
```

This implementation guide provides the concrete patterns and code structures needed to deploy the enhanced prompting framework effectively. The modular design allows for incremental implementation while maintaining safety and quality standards throughout the development process.