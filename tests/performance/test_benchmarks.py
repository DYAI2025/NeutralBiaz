"""
Performance benchmarks and load testing for the BiazNeutralize AI system.
Tests system performance under various loads and measures key metrics.
"""
import pytest
import asyncio
import time
import statistics
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'bias-engine', 'src'))

from bias_engine.core.detector import BiasDetector
from bias_engine.core.models import AnalysisRequest, BiasResult
from bias_engine.cultural.adapter import CulturalAdapter
from bias_engine.llm.client import LLMClient
from fastapi.testclient import TestClient


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self):
        self.response_times: List[float] = []
        self.throughput: float = 0.0
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.error_rate: float = 0.0
        self.concurrent_users: int = 0

    def add_response_time(self, time_seconds: float):
        self.response_times.append(time_seconds)

    def calculate_stats(self):
        if not self.response_times:
            return {}

        return {
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "avg_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": self.percentile(self.response_times, 95),
            "p99_response_time": self.percentile(self.response_times, 99),
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "avg_cpu_usage": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory_usage": statistics.mean(self.memory_usage) if self.memory_usage else 0
        }

    @staticmethod
    def percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c


class SystemMonitor:
    """Monitor system resources during testing."""

    def __init__(self):
        self.monitoring = False
        self.metrics = PerformanceMetrics()
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Monitor system resources in background."""
        process = psutil.Process(os.getpid())

        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_mb = process.memory_info().rss / 1024 / 1024

                self.metrics.cpu_usage.append(cpu_percent)
                self.metrics.memory_usage.append(memory_mb)

                time.sleep(0.5)
            except Exception:
                break


@pytest.fixture(scope="session")
def bias_detector():
    """Create bias detector for performance testing."""
    detector = BiasDetector()
    return detector


@pytest.fixture(scope="session")
def cultural_adapter():
    """Create cultural adapter for performance testing."""
    adapter = CulturalAdapter()
    return adapter


@pytest.fixture
def test_data():
    """Generate test data for performance testing."""
    return {
        "short_texts": [
            "This is clearly the best solution.",
            "Obviously this approach works perfectly.",
            "The evidence definitely supports our view."
        ] * 10,
        "medium_texts": [
            "Based on our comprehensive analysis of the available data, it becomes increasingly apparent that the proposed methodology demonstrates significant advantages over traditional approaches. The evidence strongly suggests that implementation of this solution would yield optimal results for all stakeholders involved."
        ] * 10,
        "long_texts": [
            """The extensive research conducted over the past several years has definitively established that our innovative approach represents a paradigm shift in the field. Multiple independent studies have consistently demonstrated the superiority of our methodology, with results that clearly indicate a substantial improvement in efficiency metrics across all measured parameters. The overwhelming evidence supports the conclusion that this revolutionary framework will fundamentally transform industry standards and practices, offering unprecedented benefits to organizations willing to embrace this groundbreaking innovation."""
        ] * 5,
        "bias_heavy_texts": [
            "The research obviously confirms what we already knew - our original hypothesis was completely correct from the beginning.",
            "Based on the first study we reviewed, it's clear that all subsequent findings will support the same conclusion.",
            "Since this story was prominently featured in the news recently, it's definitely a widespread and critical issue affecting everyone.",
            "All successful companies use this exact strategy, so it's guaranteed to work for any business.",
            "The senior management team has made their decision, and questioning their authority would be completely inappropriate."
        ] * 5
    }


class TestCoreDetectorPerformance:
    """Performance tests for core bias detection engine."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_single_text_latency(self, bias_detector, test_data):
        """Test latency for single text analysis."""
        await bias_detector.initialize()

        latencies = []
        text_lengths = []

        # Test different text lengths
        all_texts = (test_data["short_texts"][:5] +
                    test_data["medium_texts"][:5] +
                    test_data["long_texts"][:3])

        for text in all_texts:
            request = AnalysisRequest(text=text, cultural_context="en-US")

            start_time = time.time()
            result = await bias_detector.analyze(request)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)
            text_lengths.append(len(text))

            # Verify result quality
            assert isinstance(result, BiasResult)
            assert 0.0 <= result.overall_score <= 1.0
            assert result.processing_time > 0

        # Performance assertions
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        print(f"Average latency: {avg_latency:.3f}s")
        print(f"Maximum latency: {max_latency:.3f}s")
        print(f"Text lengths: {min(text_lengths)} - {max(text_lengths)} chars")

        # Performance targets (NFR-1: <3s response time)
        assert avg_latency < 3.0
        assert max_latency < 5.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_analysis_throughput(self, bias_detector, test_data):
        """Test throughput with concurrent analysis requests."""
        await bias_detector.initialize()

        monitor = SystemMonitor()
        monitor.start_monitoring()

        try:
            # Test with different concurrency levels
            concurrency_levels = [1, 5, 10, 20]
            results = {}

            for concurrency in concurrency_levels:
                texts = test_data["medium_texts"][:concurrency]
                requests = [AnalysisRequest(text=text, cultural_context="en-US")
                           for text in texts]

                start_time = time.time()

                # Run concurrent analyses
                tasks = [bias_detector.analyze(request) for request in requests]
                analysis_results = await asyncio.gather(*tasks)

                end_time = time.time()
                total_time = end_time - start_time
                throughput = len(analysis_results) / total_time

                results[concurrency] = {
                    "throughput": throughput,
                    "total_time": total_time,
                    "successful": len(analysis_results)
                }

                print(f"Concurrency {concurrency}: {throughput:.2f} texts/second")

            # Verify throughput scaling
            assert results[1]["throughput"] > 0
            assert results[20]["successful"] == 20  # All should succeed

            # Performance target (NFR-2: 100 requests/minute = ~1.67/second)
            assert results[10]["throughput"] >= 1.0

        finally:
            monitor.stop_monitoring()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, bias_detector, test_data):
        """Test memory usage during sustained load."""
        await bias_detector.initialize()

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process many texts to test memory stability
        texts = test_data["medium_texts"] * 10  # 100 texts

        memory_samples = []
        for i, text in enumerate(texts):
            request = AnalysisRequest(text=text, cultural_context="en-US")
            result = await bias_detector.analyze(request)

            if i % 10 == 0:  # Sample memory every 10 requests
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)

        final_memory = process.memory_info().rss / 1024 / 1024
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory

        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Maximum memory: {max_memory:.1f} MB")
        print(f"Memory growth: {memory_growth:.1f} MB")

        # Memory usage should be reasonable (NFR-3: <2GB memory usage)
        assert max_memory < 2000  # Less than 2GB
        assert memory_growth < 500  # Growth less than 500MB

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_bias_detection_accuracy_under_load(self, bias_detector, test_data):
        """Test that accuracy remains high under load."""
        await bias_detector.initialize()

        # Use texts with known bias patterns
        bias_texts = test_data["bias_heavy_texts"]

        # Test under different load conditions
        load_conditions = [
            {"concurrency": 1, "iterations": 5},
            {"concurrency": 5, "iterations": 3},
            {"concurrency": 10, "iterations": 2}
        ]

        accuracy_results = []

        for condition in load_conditions:
            all_results = []

            for iteration in range(condition["iterations"]):
                # Create concurrent requests
                requests = [AnalysisRequest(text=text, cultural_context="en-US")
                           for text in bias_texts[:condition["concurrency"]]]

                # Process concurrently
                tasks = [bias_detector.analyze(request) for request in requests]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)

            # Calculate accuracy metrics
            high_confidence_detections = sum(1 for r in all_results
                                           if r.overall_score > 0.4 and r.confidence > 0.6)
            total_analyses = len(all_results)
            accuracy = high_confidence_detections / total_analyses if total_analyses > 0 else 0

            accuracy_results.append({
                "concurrency": condition["concurrency"],
                "accuracy": accuracy,
                "total_analyses": total_analyses
            })

            print(f"Concurrency {condition['concurrency']}: {accuracy:.2%} accuracy")

        # Accuracy should remain high under load (FR-1: F1 ≥ 0.85)
        for result in accuracy_results:
            assert result["accuracy"] >= 0.75  # At least 75% should detect bias


class TestCulturalAdapterPerformance:
    """Performance tests for cultural adaptation engine."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cultural_adaptation_latency(self, cultural_adapter, test_data):
        """Test latency of cultural adaptation process."""
        await cultural_adapter.initialize()

        cultures = ["en-US", "ja-JP", "de-DE", "fr-FR", "zh-CN"]
        texts = test_data["medium_texts"][:10]

        adaptation_times = []

        for culture in cultures:
            for text in texts:
                from bias_engine.cultural.models import CulturalContext
                context = CulturalContext(culture, cultural_adapter.profiles.get(culture))

                start_time = time.time()
                result = await cultural_adapter.adapt_bias_detection(text, context, 0.5)
                end_time = time.time()

                adaptation_times.append(end_time - start_time)

        avg_adaptation_time = statistics.mean(adaptation_times)
        max_adaptation_time = max(adaptation_times)

        print(f"Average adaptation time: {avg_adaptation_time:.3f}s")
        print(f"Maximum adaptation time: {max_adaptation_time:.3f}s")

        # Performance targets for cultural adaptation
        assert avg_adaptation_time < 1.0  # Should be very fast
        assert max_adaptation_time < 2.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cross_cultural_comparison_performance(self, cultural_adapter, test_data):
        """Test performance of cross-cultural comparisons."""
        await cultural_adapter.initialize()

        texts = test_data["short_texts"][:5]
        cultures = ["en-US", "ja-JP", "de-DE"]

        comparison_times = []

        for text in texts:
            start_time = time.time()
            results = await cultural_adapter.cross_cultural_analysis(text, cultures, 0.6)
            end_time = time.time()

            comparison_times.append(end_time - start_time)
            assert len(results) == len(cultures)

        avg_comparison_time = statistics.mean(comparison_times)

        print(f"Average cross-cultural comparison time: {avg_comparison_time:.3f}s")

        # Should handle multiple cultures efficiently
        assert avg_comparison_time < 3.0


class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest.fixture
    def api_client(self):
        """Create API test client."""
        from bias_engine.api.main import create_app
        app = create_app()
        return TestClient(app)

    @pytest.mark.slow
    def test_api_endpoint_latency(self, api_client, test_data):
        """Test API endpoint response latency."""
        endpoint_tests = [
            {
                "endpoint": "/api/v1/analyze/bias",
                "method": "post",
                "data": {
                    "text": test_data["medium_texts"][0],
                    "cultural_context": "en-US",
                    "detail_level": "high"
                }
            },
            {
                "endpoint": "/api/v1/neutralize/text",
                "method": "post",
                "data": {
                    "text": test_data["bias_heavy_texts"][0],
                    "cultural_context": "en-US",
                    "preserve_style": True
                }
            }
        ]

        for test_config in endpoint_tests:
            latencies = []

            for _ in range(10):  # 10 requests per endpoint
                start_time = time.time()

                if test_config["method"] == "post":
                    response = api_client.post(test_config["endpoint"], json=test_config["data"])
                else:
                    response = api_client.get(test_config["endpoint"])

                end_time = time.time()

                assert response.status_code == 200
                latencies.append(end_time - start_time)

            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)

            print(f"{test_config['endpoint']}: avg={avg_latency:.3f}s, max={max_latency:.3f}s")

            # API performance targets (NFR-1: <3s response time)
            assert avg_latency < 3.0
            assert max_latency < 5.0

    @pytest.mark.slow
    def test_api_concurrent_load(self, api_client, test_data):
        """Test API performance under concurrent load."""
        def make_request():
            response = api_client.post("/api/v1/analyze/bias", json={
                "text": test_data["medium_texts"][0],
                "cultural_context": "en-US"
            })
            return response.status_code == 200, time.time()

        # Test with increasing concurrent users
        user_loads = [5, 10, 20]

        for num_users in user_loads:
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                start_time = time.time()

                futures = [executor.submit(make_request) for _ in range(num_users)]
                results = [future.result() for future in futures]

                end_time = time.time()
                total_time = end_time - start_time

                success_count = sum(1 for success, _ in results if success)
                success_rate = success_count / len(results)
                throughput = success_count / total_time

                print(f"Users: {num_users}, Success rate: {success_rate:.2%}, Throughput: {throughput:.2f} req/s")

                # Performance targets
                assert success_rate >= 0.95  # 95% success rate
                assert throughput >= 1.0     # At least 1 request/second

    @pytest.mark.slow
    def test_batch_processing_performance(self, api_client, test_data):
        """Test batch processing performance."""
        batch_sizes = [5, 10, 20, 50]

        for batch_size in batch_sizes:
            batch_data = {
                "texts": test_data["short_texts"][:batch_size],
                "cultural_context": "en-US"
            }

            start_time = time.time()
            response = api_client.post("/api/v1/analyze/batch", json=batch_data)
            end_time = time.time()

            if response.status_code == 200:
                total_time = end_time - start_time
                throughput = batch_size / total_time

                result_data = response.json()
                assert len(result_data["results"]) == batch_size

                print(f"Batch size: {batch_size}, Time: {total_time:.3f}s, Throughput: {throughput:.2f} texts/s")

                # Batch processing should be efficient
                assert throughput >= batch_size / 10.0  # At least 1 text per 10 seconds per batch


class TestEndToEndPerformance:
    """End-to-end performance tests simulating real user scenarios."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, bias_detector, cultural_adapter, test_data):
        """Test performance of complete analysis workflow."""
        await bias_detector.initialize()
        await cultural_adapter.initialize()

        # Simulate complete user workflow
        workflow_times = []

        for text in test_data["bias_heavy_texts"]:
            workflow_start = time.time()

            # Step 1: Bias detection
            request = AnalysisRequest(text=text, cultural_context="en-US")
            bias_result = await bias_detector.analyze(request)

            # Step 2: Cultural adaptation (if bias detected)
            if bias_result.overall_score > 0.3:
                from bias_engine.cultural.models import CulturalContext
                context = CulturalContext("en-US", cultural_adapter.profiles.get("en-US"))
                cultural_result = await cultural_adapter.adapt_bias_detection(
                    text, context, bias_result.overall_score
                )

            # Step 3: Cross-cultural comparison (if needed)
            if bias_result.overall_score > 0.5:
                comparison_result = await cultural_adapter.cross_cultural_analysis(
                    text, ["en-US", "ja-JP"], bias_result.overall_score
                )

            workflow_end = time.time()
            workflow_times.append(workflow_end - workflow_start)

        avg_workflow_time = statistics.mean(workflow_times)
        max_workflow_time = max(workflow_times)

        print(f"Average complete workflow time: {avg_workflow_time:.3f}s")
        print(f"Maximum complete workflow time: {max_workflow_time:.3f}s")

        # Complete workflow should be efficient
        assert avg_workflow_time < 5.0  # Within 5 seconds on average
        assert max_workflow_time < 10.0  # No workflow over 10 seconds

    @pytest.mark.slow
    def test_sustained_load_simulation(self, test_data):
        """Test system behavior under sustained load over time."""
        from bias_engine.api.main import create_app
        app = create_app()
        client = TestClient(app)

        monitor = SystemMonitor()
        monitor.start_monitoring()

        try:
            # Simulate sustained load for 2 minutes
            duration = 120  # seconds
            request_interval = 2  # seconds between requests

            start_time = time.time()
            request_count = 0
            successful_requests = 0

            while time.time() - start_time < duration:
                request_start = time.time()

                response = client.post("/api/v1/analyze/bias", json={
                    "text": test_data["medium_texts"][request_count % len(test_data["medium_texts"])],
                    "cultural_context": "en-US"
                })

                request_end = time.time()
                monitor.metrics.add_response_time(request_end - request_start)

                if response.status_code == 200:
                    successful_requests += 1

                request_count += 1

                # Wait for next request
                sleep_time = request_interval - (request_end - request_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_time = time.time() - start_time
            monitor.metrics.throughput = successful_requests / total_time
            monitor.metrics.error_rate = (request_count - successful_requests) / request_count

            stats = monitor.metrics.calculate_stats()

            print("Sustained Load Test Results:")
            print(f"Duration: {total_time:.1f}s")
            print(f"Total requests: {request_count}")
            print(f"Successful requests: {successful_requests}")
            print(f"Throughput: {stats['throughput']:.2f} req/s")
            print(f"Error rate: {stats['error_rate']:.2%}")
            print(f"Average response time: {stats['avg_response_time']:.3f}s")
            print(f"95th percentile response time: {stats['p95_response_time']:.3f}s")
            print(f"Average CPU usage: {stats['avg_cpu_usage']:.1f}%")
            print(f"Average memory usage: {stats['avg_memory_usage']:.1f} MB")

            # System should remain stable under sustained load
            assert stats["error_rate"] < 0.05  # Less than 5% error rate
            assert stats["avg_response_time"] < 5.0  # Reasonable response times
            assert stats["p95_response_time"] < 10.0  # 95% under 10 seconds
            assert stats["avg_cpu_usage"] < 80.0  # CPU usage reasonable

        finally:
            monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])