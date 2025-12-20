# Public API Inventory
Total APIs: 978

## udl_rating_framework.analytics.bi_exporter

### Classes
- `BIDataset(name: str, description: str, data: pandas.core.frame.DataFrame, metadata: Dict[str, Any], export_timestamp: datetime.datetime) -> None` [✓]
- `BIExportConfig(format: str, include_metadata: bool = True, include_trends: bool = True, include_forecasts: bool = False, aggregation_level: str = 'detailed', time_grouping: str = 'daily') -> None` [✓]
- `BusinessIntelligenceExporter()` [✓]

## udl_rating_framework.analytics.improvement_advisor

### Classes
- `ImprovementAdvisor()` [✓]
- `ImprovementPlan(udl_file: str, current_score: float, target_score: float, suggestions: List[udl_rating_framework.analytics.improvement_advisor.ImprovementSuggestion], implementation_order: List[str], estimated_timeline: str, success_metrics: List[str]) -> None` [✓]
- `ImprovementSuggestion(category: str, priority: str, title: str, description: str, rationale: str, expected_impact: float, effort_level: str, specific_actions: List[str], related_metrics: List[str]) -> None` [✓]

## udl_rating_framework.analytics.portfolio_analyzer

### Classes
- `BenchmarkAnalysis(percentiles: Dict[str, Dict[str, float]], industry_standards: Dict[str, float], performance_gaps: Dict[str, Dict[str, float]], improvement_potential: Dict[str, float]) -> None` [✓]
- `PortfolioAnalyzer(min_reports_per_project: int = 5, clustering_method: str = 'kmeans', n_clusters: int = 3)` [✓]
- `PortfolioComparison(project_profiles: Dict[str, udl_rating_framework.analytics.portfolio_analyzer.ProjectProfile], rankings: Dict[str, List[str]], clusters: Dict[str, List[str]], correlations: pandas.core.frame.DataFrame, outliers: List[str], recommendations: Dict[str, List[str]]) -> None` [✓]
- `ProjectProfile(project_name: str, udl_files: List[str], total_reports: int, avg_overall_score: float, std_overall_score: float, avg_confidence: float, metric_averages: Dict[str, float], quality_trend: float, consistency_score: float, maturity_level: str, risk_level: str) -> None` [✓]

## udl_rating_framework.analytics.time_series_analyzer

### Classes
- `QualityEvolution(metric_name: str, time_series: pandas.core.series.Series, trend_analysis: udl_rating_framework.analytics.time_series_analyzer.TimeSeriesMetrics, forecast: Optional[pandas.core.series.Series], confidence_intervals: Optional[Tuple[pandas.core.series.Series, pandas.core.series.Series]], summary: str) -> None` [✓]
- `TimeSeriesAnalyzer(min_observations: int = 10, anomaly_threshold: float = 2.0, seasonality_periods: List[int] = None)` [✓]
- `TimeSeriesMetrics(trend_slope: float, trend_p_value: float, seasonality_strength: float, volatility: float, autocorrelation: float, change_points: List[datetime.datetime], anomalies: List[Tuple[datetime.datetime, float, str]]) -> None` [✓]

## udl_rating_framework.analytics.trend_predictor

### Classes
- `PredictionResult(predictions: pandas.core.series.Series, confidence_intervals: Tuple[pandas.core.series.Series, pandas.core.series.Series], model_performance: Dict[str, float], feature_importance: Optional[Dict[str, float]], prediction_horizon: int, model_type: str) -> None` [✓]
- `TrendAnalysis(historical_trends: Dict[str, float], seasonal_patterns: Dict[str, Dict[str, float]], volatility_metrics: Dict[str, float], regime_changes: Dict[str, List[datetime.datetime]], forecast_accuracy: Dict[str, float]) -> None` [✓]
- `TrendPredictor(prediction_horizon: int = 30, validation_split: float = 0.2, models: List[str] = None)` [✓]

## udl_rating_framework.benchmarks.performance_benchmarks

### Classes
- `BenchmarkResult(name: str, execution_time: float, memory_usage_mb: float, cpu_usage_percent: float, input_size: int, throughput: float, metadata: Dict[str, Any] = <factory>) -> None` [✓]
- `BenchmarkSuite(suite_name: str, results: List[udl_rating_framework.benchmarks.performance_benchmarks.BenchmarkResult] = <factory>, total_time: float = 0.0, timestamp: float = <factory>) -> None` [✓]
- `PerformanceBenchmark(output_dir: Optional[pathlib._local.Path] = None)` [✓]

### Functions
- `run_comprehensive_benchmarks(output_dir: Optional[pathlib._local.Path] = None) -> Dict[str, udl_rating_framework.benchmarks.performance_benchmarks.BenchmarkSuite]` [✓]

## udl_rating_framework.cli.commands.analytics

### Functions
- `load_reports_from_directory(reports_dir: pathlib._local.Path) -> List[udl_rating_framework.core.pipeline.QualityReport]` [✓]

## udl_rating_framework.cli.commands.integration

### Functions
- `add_integration_commands(cli)` [✓]

## udl_rating_framework.cli.config

### Classes
- `Config(metrics: udl_rating_framework.cli.config.MetricConfig = <factory>, training: udl_rating_framework.cli.config.TrainingConfig = <factory>, model: udl_rating_framework.cli.config.ModelConfig = <factory>, evaluation: udl_rating_framework.cli.config.EvaluationConfig = <factory>, output: udl_rating_framework.cli.config.OutputConfig = <factory>) -> None` [✓]
- `EvaluationConfig(k_folds: int = 5, bootstrap_samples: int = 1000, confidence_level: float = 0.95, calibration_bins: int = 10) -> None` [✓]
- `MetricConfig(consistency_weight: float = 0.25, completeness_weight: float = 0.25, expressiveness_weight: float = 0.25, structural_coherence_weight: float = 0.25) -> None` [✓]
- `ModelConfig(d_model: int = 256, d_input: int = 64, iterations: int = 20, n_synch_out: int = 32, vocab_size: int = 10000, max_sequence_length: int = 512) -> None` [✓]
- `OutputConfig(formats: list = <factory>, output_dir: str = 'output', include_traces: bool = True, precision: int = 6) -> None` [✓]
- `TrainingConfig(batch_size: int = 32, learning_rate: float = 0.001, epochs: int = 100, validation_split: float = 0.2, early_stopping_patience: int = 10, checkpoint_dir: str = 'checkpoints', alpha: float = 0.7, beta: float = 0.3) -> None` [✓]

### Functions
- `create_default_config() -> udl_rating_framework.cli.config.Config` [✓]
- `load_config(config_path: pathlib._local.Path) -> Dict[str, Any]` [✓]
- `save_config(config: udl_rating_framework.cli.config.Config, config_path: pathlib._local.Path) -> None` [✓]
- `validate_config(config_dict: Dict[str, Any]) -> None` [✓]

## udl_rating_framework.cli.main

### Functions
- `main()` [✓]

## udl_rating_framework.core.aggregation

### Classes
- `MetricAggregator(weights: Dict[str, float])` [✓]

## udl_rating_framework.core.caching

### Classes
- `CacheEntry(data: Any, created_at: datetime.datetime, access_count: int = 0, last_accessed: Optional[datetime.datetime] = None, file_hash: Optional[str] = None, file_mtime: Optional[float] = None) -> None` [✓]
- `LRUCache(max_size: int = 1000)` [✓]
- `MetricCache(max_size: int = 1000, ttl_hours: int = 12)` [✓]
- `UDLRepresentationCache(max_size: int = 500, ttl_hours: int = 24)` [✓]

### Functions
- `clear_all_caches() -> None` [✓]
- `get_cache_stats() -> Dict[str, Any]` [✓]
- `get_metric_cache() -> udl_rating_framework.core.caching.MetricCache` [✓]
- `get_udl_cache() -> udl_rating_framework.core.caching.UDLRepresentationCache` [✓]

## udl_rating_framework.core.confidence

### Classes
- `ConfidenceCalculator()` [✓]

## udl_rating_framework.core.distributed

### Classes
- `DistributedBatchProcessor(backend: str = 'auto', cluster_address: Optional[str] = None, num_workers: Optional[int] = None)` [✓]
- `DistributedConfig(backend: str = 'ray', cluster_address: Optional[str] = None, num_workers: Optional[int] = None, resources_per_worker: Dict[str, float] = None, batch_size: int = 10, max_retries: int = 3, timeout_seconds: float = 300.0, enable_progress_tracking: bool = True, checkpoint_interval: int = 100, result_serialization: str = 'pickle') -> None` [✓]
- `DistributedProcessor(config: Optional[udl_rating_framework.core.distributed.DistributedConfig] = None)` [✓]
- `DistributedResult(task_id: str, success: bool, result: Optional[udl_rating_framework.core.pipeline.QualityReport] = None, error: Optional[str] = None, processing_time: float = 0.0, worker_id: Optional[str] = None, node_id: Optional[str] = None) -> None` [✓]
- `DistributedTask(task_id: str, file_path: str, content: Optional[str] = None, metric_names: List[str] = None, weights: Optional[Dict[str, float]] = None, priority: int = 0, retry_count: int = 0) -> None` [✓]

### Functions
- `<lambda>(f)` [✗]
- `process_directory_distributed(directory: pathlib._local.Path, metric_names: List[str], backend: str = 'auto', cluster_address: Optional[str] = None, num_workers: Optional[int] = None, weights: Optional[Dict[str, float]] = None, file_patterns: List[str] = None, show_progress: bool = True) -> tuple` [✓]
- `process_files_distributed(file_paths: List[pathlib._local.Path], metric_names: List[str], backend: str = 'auto', cluster_address: Optional[str] = None, num_workers: Optional[int] = None, weights: Optional[Dict[str, float]] = None, show_progress: bool = True) -> tuple` [✓]
- `process_udl_task_dask(task: udl_rating_framework.core.distributed.DistributedTask) -> udl_rating_framework.core.distributed.DistributedResult` [✓]
- `<lambda>(*args, **kwargs)` [✗]

## udl_rating_framework.core.gpu_acceleration

### Classes
- `GPUAcceleratedCTM(model: udl_rating_framework.models.ctm_adapter.UDLRatingCTM, device_manager: Optional[udl_rating_framework.core.gpu_acceleration.GPUDeviceManager] = None, batch_size: int = 32, max_sequence_length: int = 512, enable_mixed_precision: bool = True, enable_compilation: bool = True)` [✓]
- `GPUAcceleratedProcessor(model_config: Optional[Dict[str, Any]] = None, batch_size: int = 32, max_sequence_length: int = 512, enable_mixed_precision: bool = True)` [✓]
- `GPUDeviceManager()` [✓]
- `UDLDataset(udl_representations: List[udl_rating_framework.core.representation.UDLRepresentation], vocabulary: udl_rating_framework.models.ctm_adapter.UDLTokenVocabulary, max_length: int = 512)` [✓]

### Functions
- `benchmark_gpu_performance(test_udls: List[udl_rating_framework.core.representation.UDLRepresentation], batch_sizes: List[int] = None, sequence_lengths: List[int] = None) -> Dict[str, Any]` [✓]
- `process_files_gpu(file_paths: List[pathlib._local.Path], batch_size: int = 32, max_sequence_length: int = 512, enable_mixed_precision: bool = True, model_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]` [✓]
- `process_udls_gpu(udl_representations: List[udl_rating_framework.core.representation.UDLRepresentation], batch_size: int = 32, max_sequence_length: int = 512, enable_mixed_precision: bool = True, model_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]` [✓]

## udl_rating_framework.core.incremental

### Classes
- `ChangeSet(file_path: str, change_type: str, old_snapshot: Optional[udl_rating_framework.core.incremental.FileSnapshot] = None, new_snapshot: Optional[udl_rating_framework.core.incremental.FileSnapshot] = None, affected_lines: Set[int] = <factory>, affected_constructs: Set[str] = <factory>, change_summary: Dict[str, Any] = <factory>) -> None` [✓]
- `DependencyTracker()` [✓]
- `FileSnapshot(file_path: str, content_hash: str, modification_time: float, file_size: int, line_count: int, checksum: str, metadata: Dict[str, Any] = <factory>) -> None` [✓]
- `IncrementalCache(cache_dir: Optional[pathlib._local.Path] = None)` [✓]
- `IncrementalProcessor(cache_dir: Optional[pathlib._local.Path] = None, metric_names: List[str] = None)` [✓]
- `IncrementalResult(file_path: str, was_cached: bool, was_incremental: bool, change_ratio: float, processing_time: float, cache_hit_ratio: float, recomputed_metrics: Set[str], report: udl_rating_framework.core.pipeline.QualityReport, change_set: Optional[udl_rating_framework.core.incremental.ChangeSet] = None) -> None` [✓]

### Functions
- `process_udl_directory_incremental(directory: pathlib._local.Path, metric_names: List[str] = None, cache_dir: Optional[pathlib._local.Path] = None, file_patterns: List[str] = None) -> List[udl_rating_framework.core.incremental.IncrementalResult]` [✓]
- `process_udl_incremental(file_path: pathlib._local.Path, metric_names: List[str] = None, cache_dir: Optional[pathlib._local.Path] = None) -> udl_rating_framework.core.incremental.IncrementalResult` [✓]

## udl_rating_framework.core.memory_mapping

### Classes
- `MemoryMapConfig(access_mode: int = 1, page_size: int = 4096, prefetch_size: int = 65536, cache_size: int = 100, enable_sequential_hint: bool = True, enable_random_hint: bool = False, enable_willneed_hint: bool = True, thread_safe: bool = True) -> None` [✓]
- `MemoryMappedFile(file_path: pathlib._local.Path, config: Optional[udl_rating_framework.core.memory_mapping.MemoryMapConfig] = None)` [✓]
- `MemoryMappedUDLProcessor(config: Optional[udl_rating_framework.core.memory_mapping.MemoryMapConfig] = None, metric_names: List[str] = None)` [✓]

### Functions
- `analyze_udl_file_structure(file_path: pathlib._local.Path) -> Dict[str, Any]` [✓]
- `process_large_udl_file(file_path: pathlib._local.Path, metric_names: List[str] = None, chunk_size: int = 1048576, overlap_size: int = 1024) -> udl_rating_framework.core.pipeline.QualityReport` [✓]

## udl_rating_framework.core.metrics.base

### Classes
- `MetricRegistry() -> 'MetricRegistry'` [✓]
- `QualityMetric()` [✓]

## udl_rating_framework.core.metrics.completeness

### Classes
- `CompletenessMetric()` [✓]
- `Construct(name: str, construct_type: str, metadata: Dict[str, Any] = None)` [✓]

## udl_rating_framework.core.metrics.consistency

### Classes
- `ConsistencyMetric()` [✓]

## udl_rating_framework.core.metrics.cross_language_compatibility

### Classes
- `CrossLanguageCompatibilityMetric()` [✓]

## udl_rating_framework.core.metrics.evolution_tracking

### Classes
- `EvolutionTrackingMetric()` [✓]

## udl_rating_framework.core.metrics.expressiveness

### Classes
- `ExpressivenessMetric()` [✓]
- `Grammar(rules: List[udl_rating_framework.core.representation.GrammarRule])` [✓]

## udl_rating_framework.core.metrics.maintainability

### Classes
- `MaintainabilityMetric()` [✓]

## udl_rating_framework.core.metrics.readability

### Classes
- `ReadabilityMetric()` [✓]

## udl_rating_framework.core.metrics.semantic_similarity

### Classes
- `SemanticSimilarityMetric(use_pretrained: bool = False)` [✓]

## udl_rating_framework.core.metrics.structural_coherence

### Classes
- `StructuralCoherenceMetric()` [✓]

## udl_rating_framework.core.multiprocessing

### Classes
- `BatchProcessingStats(total_files: int, successful: int, failed: int, total_time: float, average_time_per_file: float, max_time: float, min_time: float, worker_count: int) -> None` [✓]
- `ParallelProcessor(max_workers: Optional[int] = None, chunk_size: int = 1)` [✓]
- `ProcessingResult(success: bool, result: Any = None, error: Optional[str] = None, processing_time: float = 0.0, worker_id: Optional[int] = None) -> None` [✓]

### Functions
- `compute_metrics_for_udls_parallel(udl_representations: List[udl_rating_framework.core.representation.UDLRepresentation], metric_names: List[str], max_workers: Optional[int] = None, show_progress: bool = True) -> Dict[str, List[Tuple[str, Union[float, str]]]]` [✓]
- `create_progress_callback(description: str = 'Processing') -> Callable[[int, int], NoneType]` [✓]
- `process_udl_files_parallel(file_paths: List[Union[str, pathlib._local.Path]], metric_names: List[str], max_workers: Optional[int] = None, weights: Optional[Dict[str, float]] = None, enable_tracing: bool = False, show_progress: bool = True) -> Tuple[List[udl_rating_framework.core.pipeline.QualityReport], udl_rating_framework.core.multiprocessing.BatchProcessingStats]` [✓]

## udl_rating_framework.core.performance

### Classes
- `PerformanceConfig(strategy: udl_rating_framework.core.performance.ProcessingStrategy = <ProcessingStrategy.AUTO: 'auto'>, fallback_strategy: udl_rating_framework.core.performance.ProcessingStrategy = <ProcessingStrategy.PARALLEL: 'parallel'>, max_memory_mb: int = 1024, max_workers: Optional[int] = None, max_file_size_mb: int = 100, distributed_backend: str = 'auto', cluster_address: Optional[str] = None, enable_gpu: bool = True, gpu_batch_size: int = 32, gpu_max_sequence_length: int = 512, streaming_chunk_size: int = 1048576, streaming_overlap_size: int = 1024, memory_map_threshold_mb: int = 50, memory_map_page_size: int = 4096, enable_incremental: bool = True, incremental_cache_dir: Optional[pathlib._local.Path] = None, enable_caching: bool = True, cache_ttl_hours: int = 24, enable_progress: bool = True, progress_callback: Optional[Callable[[int, int], NoneType]] = None) -> None` [✓]
- `PerformanceOptimizer(config: Optional[udl_rating_framework.core.performance.PerformanceConfig] = None)` [✓]
- `PerformanceResult(strategy_used: udl_rating_framework.core.performance.ProcessingStrategy, total_files: int, successful_files: int, failed_files: int, processing_time: float, throughput: float, memory_peak_mb: float, cache_hit_ratio: float, reports: List[udl_rating_framework.core.pipeline.QualityReport], errors: List[str] = <factory>, metadata: Dict[str, Any] = <factory>) -> None` [✓]
- `ProcessingStrategy(*values)` [✓]

### Functions
- `benchmark_all_strategies(test_files: List[pathlib._local.Path], metric_names: List[str], iterations: int = 3) -> Dict[str, Dict[str, float]]` [✓]
- `process_directory_optimized(directory: pathlib._local.Path, metric_names: List[str], weights: Optional[Dict[str, float]] = None, file_patterns: List[str] = None, config: Optional[udl_rating_framework.core.performance.PerformanceConfig] = None) -> udl_rating_framework.core.performance.PerformanceResult` [✓]
- `process_files_optimized(file_paths: List[pathlib._local.Path], metric_names: List[str], weights: Optional[Dict[str, float]] = None, config: Optional[udl_rating_framework.core.performance.PerformanceConfig] = None) -> udl_rating_framework.core.performance.PerformanceResult` [✓]

## udl_rating_framework.core.pipeline

### Classes
- `ComputationStep(step_number: int, operation: str, formula: str, inputs: Dict[str, Any], output: Any, intermediate_values: Dict[str, float] = <factory>) -> None` [✓]
- `QualityReport(overall_score: float, confidence: float, metric_scores: Dict[str, float], metric_formulas: Dict[str, str], computation_trace: List[udl_rating_framework.core.pipeline.ComputationStep], error_bounds: Dict[str, Tuple[float, float]], timestamp: datetime.datetime, udl_file: str, errors: List[str] = <factory>, warnings: List[str] = <factory>) -> None` [✓]
- `RatingPipeline(metric_names: List[str], weights: Optional[Dict[str, float]] = None, enable_tracing: bool = True, enable_caching: bool = True)` [✓]

## udl_rating_framework.core.representation

### Classes
- `AST(node_type: str, value: Any = None, children: List[ForwardRef('AST')] = None)` [✓]
- `Constraint(type: str, condition: str, metadata: Tuple[Tuple[str, Any], ...] = ()) -> None` [✓]
- `GrammarFormat(*values)` [✓]
- `GrammarRule(lhs: str, rhs: List[str], constraints: List[udl_rating_framework.core.representation.Constraint], metadata: Dict[str, Any]) -> None` [✓]
- `Token(text: str, type: udl_rating_framework.core.representation.TokenType, position: int, line: int, column: int) -> None` [✓]
- `TokenType(*values)` [✓]
- `UDLRepresentation(source_text: str, file_path: str)` [✓]
- `UDLTokenizer()` [✓]

## udl_rating_framework.core.streaming

### Classes
- `AsyncStreamingProcessor(config: Optional[udl_rating_framework.core.streaming.StreamingConfig] = None, metric_names: List[str] = None, max_concurrent_chunks: int = 4)` [✓]
- `MemoryMappedFileReader(file_path: pathlib._local.Path, encoding: str = 'utf-8')` [✓]
- `StreamingChunk(chunk_id: int, start_offset: int, end_offset: int, content: str, overlap_before: str = '', overlap_after: str = '', is_complete: bool = True, metadata: Dict[str, Any] = <factory>) -> None` [✓]
- `StreamingChunker(config: udl_rating_framework.core.streaming.StreamingConfig)` [✓]
- `StreamingConfig(chunk_size: int = 1048576, overlap_size: int = 1024, max_memory_usage: int = 104857600, buffer_size: int = 10, enable_caching: bool = True, cache_chunk_results: bool = True, progress_callback: Optional[Callable[[int, int], NoneType]] = None, error_handling: str = 'continue', encoding: str = 'utf-8', line_ending: str = '\n') -> None` [✓]
- `StreamingProcessor(config: Optional[udl_rating_framework.core.streaming.StreamingConfig] = None, metric_names: List[str] = None)` [✓]
- `StreamingResult(chunk_id: int, file_path: str, chunk_start: int, chunk_end: int, partial_results: List[udl_rating_framework.core.pipeline.QualityReport], aggregated_score: float, confidence: float, processing_time: float, error: Optional[str] = None, success: bool = True) -> None` [✓]

### Functions
- `aggregate_streaming_results(results: List[udl_rating_framework.core.streaming.StreamingResult]) -> Optional[udl_rating_framework.core.pipeline.QualityReport]` [✓]
- `process_large_file(file_path: pathlib._local.Path, metric_names: List[str] = None, chunk_size: int = 1048576, overlap_size: int = 1024, enable_async: bool = False, progress_callback: Optional[Callable[[int, int], NoneType]] = None) -> Iterator[udl_rating_framework.core.streaming.StreamingResult]` [✓]

## udl_rating_framework.evaluation.comparison

### Classes
- `ComparisonEngine(alpha: float = 0.05, bootstrap_samples: int = 1000)` [✓]
- `ComparisonResult(udl1_name: str, udl2_name: str, score1: float, score2: float, difference: float, ttest_statistic: float, ttest_pvalue: float, wilcoxon_statistic: Optional[float], wilcoxon_pvalue: Optional[float], cohens_d: float, is_significant: bool, effect_size_interpretation: str) -> None` [✓]
- `ComparisonSummary(pairwise_results: List[udl_rating_framework.evaluation.comparison.ComparisonResult], rankings: List[udl_rating_framework.evaluation.comparison.RankingResult], total_comparisons: int, significant_comparisons: int, mean_effect_size: float, score_statistics: Dict[str, float]) -> None` [✓]
- `RankingResult(udl_name: str, score: float, rank: int, confidence_interval: Tuple[float, float], rank_confidence_interval: Tuple[int, int]) -> None` [✓]

## udl_rating_framework.evaluation.evaluation_suite

### Classes
- `EvaluationResult(pearson_correlation: float, pearson_ci: Tuple[float, float], spearman_correlation: float, spearman_ci: Tuple[float, float], calibration_error: float, shapiro_statistic: float, shapiro_p_value: float, bootstrap_metrics: Dict[str, Tuple[float, float]], cv_scores: List[float], mean_cv_score: float, std_cv_score: float) -> None` [✓]
- `EvaluationSuite(k_folds: int = 5, bootstrap_samples: int = 1000, confidence_level: float = 0.95)` [✓]

## udl_rating_framework.integration.batch_processor

### Classes
- `BatchConfig(max_workers: int = 14, chunk_size: int = 10, timeout_per_file: float = 30.0, memory_limit_mb: int = 1024, enable_caching: bool = True, cache_dir: Optional[pathlib._local.Path] = None, progress_callback: Optional[Callable[[int, int], NoneType]] = None, error_handling: str = 'continue', max_retries: int = 3, output_format: str = 'json', include_detailed_metrics: bool = True, generate_summary: bool = True) -> None` [✓]
- `BatchProcessor(config: Optional[udl_rating_framework.integration.batch_processor.BatchConfig] = None)` [✓]
- `BatchResult(total_files: int, processed_files: int, failed_files: int, processing_time: float, average_quality: float, quality_distribution: Dict[str, int], file_results: Dict[str, Dict[str, Any]], errors: Dict[str, str], summary_stats: Dict[str, Any]) -> None` [✓]
- `ProcessingTask(file_path: pathlib._local.Path, content: Optional[str] = None, priority: int = 0, retry_count: int = 0) -> None` [✓]

### Functions
- `main()` [✓]

## udl_rating_framework.integration.cicd

### Classes
- `CICDConfig(min_quality_threshold: float = 0.7, fail_on_quality_drop: bool = True, generate_reports: bool = True, report_format: str = 'json', artifact_retention_days: int = 30, parallel_jobs: int = 4, timeout_minutes: int = 30) -> None` [✓]
- `CICDIntegration(config: Optional[udl_rating_framework.integration.cicd.CICDConfig] = None)` [✓]

### Functions
- `main()` [✓]

## udl_rating_framework.integration.git_hooks

### Classes
- `GitHookManager(repo_path: pathlib._local.Path, min_quality_threshold: float = 0.7, config_file: Optional[pathlib._local.Path] = None)` [✓]

### Functions
- `main()` [✓]

## udl_rating_framework.integration.ide_plugin

### Classes
- `IDEPluginManager(config: Optional[udl_rating_framework.integration.ide_plugin.PluginConfig] = None)` [✓]
- `PluginConfig(enable_real_time_checking: bool = True, quality_threshold: float = 0.7, show_detailed_metrics: bool = True, auto_save_reports: bool = False, report_directory: Optional[pathlib._local.Path] = None, update_interval: float = 1.0, max_file_size_mb: int = 10) -> None` [✓]

### Functions
- `main()` [✓]

## udl_rating_framework.integration.lsp_server

### Classes
- `Diagnostic(range: udl_rating_framework.integration.lsp_server.Range, severity: int, code: Optional[str], source: str, message: str, related_information: Optional[List[Any]] = None) -> None` [✓]
- `LSPServer(language_server: udl_rating_framework.integration.lsp_server.UDLLanguageServer)` [✓]
- `Position(line: int, character: int) -> None` [✓]
- `Range(start: udl_rating_framework.integration.lsp_server.Position, end: udl_rating_framework.integration.lsp_server.Position) -> None` [✓]
- `TextDocumentItem(uri: str, language_id: str, version: int, text: str) -> None` [✓]
- `UDLLanguageServer(min_quality_threshold: float = 0.7, enable_real_time: bool = True, debounce_delay: float = 0.5)` [✓]

### Functions
- `main()` [✓]

## udl_rating_framework.io.file_discovery

### Classes
- `FileDiscovery(supported_extensions: Optional[Set[str]] = None, case_sensitive: bool = False)` [✓]
- `FileDiscoveryError()` [✓]
- `FileDiscoveryResult(discovered_files: List[pathlib._local.Path], errors: List[str], total_directories_scanned: int, total_files_examined: int) -> None` [✓]

## udl_rating_framework.io.input_validation

### Classes
- `InputValidator(max_file_size: int = 10485760, encoding: str = 'utf-8')` [✓]
- `UDLFormat(*values)` [✓]
- `ValidationError()` [✓]
- `ValidationResult(is_valid: bool, format_type: udl_rating_framework.io.input_validation.UDLFormat, errors: List[str], warnings: List[str], line_count: int, character_count: int, detected_constructs: Set[str]) -> None` [✓]

## udl_rating_framework.io.report_generator

### Classes
- `ReportGenerator(include_visualizations: bool = True)` [✓]

## udl_rating_framework.models.ctm_adapter

### Classes
- `TrackingData(pre_activations: numpy.ndarray, post_activations: numpy.ndarray, synch_out: numpy.ndarray, synch_action: numpy.ndarray, attention_weights: numpy.ndarray, iterations: int, batch_size: int, seq_len: int, n_neurons: int, n_synch_out: int, n_synch_action: int, n_heads: int) -> None` [✓]
- `UDLRatingCTM(vocab_size: int, d_model: int = 256, d_input: int = 64, iterations: int = 20, n_synch_out: int = 32, heads: int = 8, n_synch_action: int = 16, synapse_depth: int = 3, memory_length: int = 10, deep_nlms: bool = True, memory_hidden_dims: int = 128, do_layernorm_nlm: bool = False, dropout: float = 0.1, neuron_select_type: str = 'random-pairing', n_random_pairing_self: int = 0, **ctm_kwargs)` [✓]
- `UDLTokenVocabulary()` [✓]

### Functions
- `create_udl_rating_model(vocab_size: int, **kwargs) -> udl_rating_framework.models.ctm_adapter.UDLRatingCTM` [✓]

## udl_rating_framework.training.active_learning

### Classes
- `ActiveLearner(model: udl_rating_framework.models.ctm_adapter.UDLRatingCTM, unlabeled_dataset: torch.utils.data.dataset.Dataset, metrics: List[Any], aggregator: Any, config: udl_rating_framework.training.active_learning.ActiveLearningConfig, oracle_fn: Optional[Callable] = None, device: Optional[torch.device] = None)` [✓]
- `ActiveLearningConfig(query_strategy: str = 'uncertainty_sampling', initial_pool_size: int = 100, query_batch_size: int = 20, max_iterations: int = 10, uncertainty_method: str = 'entropy', diversity_method: str = 'kmeans', n_clusters: int = 10, uncertainty_weight: float = 0.7, diversity_weight: float = 0.3, retrain_epochs: int = 10, validation_split: float = 0.2) -> None` [✓]
- `CTMUncertaintySampling(method: str = 'synchronization_entropy', device: Optional[torch.device] = None, use_temporal_analysis: bool = True)` [✓]
- `DiversitySampling(method: str = 'kmeans', n_clusters: int = 10, device: Optional[torch.device] = None)` [✓]
- `HybridSampling(uncertainty_weight: float = 0.7, diversity_weight: float = 0.3, uncertainty_method: str = 'entropy', diversity_method: str = 'kmeans', device: Optional[torch.device] = None)` [✓]
- `QueryStrategy()` [✓]

### Functions
- `create_active_learner(model: udl_rating_framework.models.ctm_adapter.UDLRatingCTM, unlabeled_dataset: torch.utils.data.dataset.Dataset, metrics: List[Any], aggregator: Any, query_strategy: str = 'uncertainty_sampling', **config_kwargs) -> udl_rating_framework.training.active_learning.ActiveLearner` [✓]

## udl_rating_framework.training.ensemble_methods

### Classes
- `CTMEnsembleTrainer(vocab_size: int, base_config: Dict[str, Any], device: Optional[torch.device] = None)` [✓]
- `EnsembleMember(model: udl_rating_framework.models.ctm_adapter.UDLRatingCTM, weight: float = 1.0, training_history: Optional[Dict[str, List[float]]] = None, model_id: Optional[str] = None) -> None` [✓]
- `EnsemblePredictor(members: List[udl_rating_framework.training.ensemble_methods.EnsembleMember], method: str = 'weighted_average', device: Optional[torch.device] = None)` [✓]

### Functions
- `create_bootstrap_ensemble(model_factory: <built-in function callable>, train_dataloader: torch.utils.data.dataloader.DataLoader, val_dataloader: torch.utils.data.dataloader.DataLoader, num_members: int = 5, bootstrap_ratio: float = 0.8, num_epochs: int = 50) -> udl_rating_framework.training.ensemble_methods.EnsemblePredictor` [✓]

## udl_rating_framework.training.hyperparameter_optimization

### Classes
- `CTMHyperparameterOptimizer(vocab_size: int, metrics: List[udl_rating_framework.core.metrics.base.QualityMetric], aggregator: udl_rating_framework.core.aggregation.MetricAggregator, param_space: udl_rating_framework.training.hyperparameter_optimization.CTMHyperparameterSpace, device: Optional[torch.device] = None)` [✓]
- `CTMHyperparameterSpace(d_model: Any = <factory>, d_input: Any = <factory>, iterations: Any = <factory>, n_synch_out: Any = <factory>, n_synch_action: Any = <factory>, neuron_select_type: Any = <factory>, n_random_pairing_self: Any = <factory>, memory_length: Any = <factory>, deep_nlms: Any = <factory>, memory_hidden_dims: Any = <factory>, do_layernorm_nlm: Any = <factory>, synapse_depth: Any = <factory>, heads: Any = <factory>, dropout: Any = <factory>, dropout_nlm: Any = <factory>, learning_rate: Any = <factory>, batch_size: Any = <factory>, alpha: Any = <factory>, beta: Any = <factory>, weight_decay: Any = <factory>) -> None` [✓]
- `OptimizationResult(best_params: Dict[str, Any], best_score: float, best_model_path: Optional[str], optimization_history: List[Dict[str, Any]], total_trials: int, optimization_time: float) -> None` [✓]

### Functions
- `create_ctm_param_space() -> udl_rating_framework.training.hyperparameter_optimization.CTMHyperparameterSpace` [✓]
- `create_focused_ctm_param_space(focus: str = 'synchronization') -> udl_rating_framework.training.hyperparameter_optimization.CTMHyperparameterSpace` [✓]
- `optimize_ctm_hyperparameters(vocab_size: int, metrics: List[udl_rating_framework.core.metrics.base.QualityMetric], aggregator: udl_rating_framework.core.aggregation.MetricAggregator, train_dataloader: torch.utils.data.dataloader.DataLoader, val_dataloader: torch.utils.data.dataloader.DataLoader, method: str = 'random', num_trials: int = 50, param_space: Optional[udl_rating_framework.training.hyperparameter_optimization.CTMHyperparameterSpace] = None, **kwargs) -> udl_rating_framework.training.hyperparameter_optimization.OptimizationResult` [✓]

## udl_rating_framework.training.training_pipeline

### Classes
- `TrainingPipeline(model: udl_rating_framework.models.ctm_adapter.UDLRatingCTM, metrics: List[udl_rating_framework.core.metrics.base.QualityMetric], aggregator: udl_rating_framework.core.aggregation.MetricAggregator, alpha: float = 0.7, beta: float = 0.3, learning_rate: float = 0.001, device: Optional[torch.device] = None)` [✓]
- `UDLDataset(udl_representations: List[udl_rating_framework.core.representation.UDLRepresentation], vocab: udl_rating_framework.models.ctm_adapter.UDLTokenVocabulary, max_length: int = 512)` [✓]

## udl_rating_framework.training.transfer_learning

### Classes
- `AdapterLayer(input_dim: int, adapter_dim: int, dropout: float = 0.1)` [✓]
- `CTMTransferConfig(source_model_path: Optional[str] = None, freeze_source_ctm: bool = True, transfer_method: str = 'temporal_dynamics', transfer_iterations: bool = True, transfer_memory_patterns: bool = True, transfer_synchronization: bool = True, fine_tune_nlms: bool = False, fine_tune_synapses: bool = True, fine_tune_synchronization: bool = True, adaptation_epochs: int = 15, fine_tune_epochs: int = 25, adaptation_lr: float = 0.0001, fine_tune_lr: float = 1e-05, preserve_temporal_structure: bool = True, adapt_memory_length: bool = False) -> None` [✓]
- `CTMTransferLearningModel(vocab_size: int, transfer_config: udl_rating_framework.training.transfer_learning.CTMTransferConfig, ctm_config: Dict[str, Any])` [✓]
- `CTMTransferLearningTrainer(vocab_size: int, transfer_config: udl_rating_framework.training.transfer_learning.CTMTransferConfig, ctm_config: Dict[str, Any], device: Optional[torch.device] = None)` [✓]
- `PretrainedFeatureExtractor(model_name: str = 'bert-base-uncased', freeze_pretrained: bool = True, fine_tune_layers: int = 2, adaptation_method: str = 'feature_extraction', adapter_dim: int = 64, projection_dim: int = 256)` [✓]

### Functions
- `create_ctm_transfer_learning_model(vocab_size: int, source_model_path: Optional[str] = None, ctm_config: Optional[Dict[str, Any]] = None, transfer_config: Optional[udl_rating_framework.training.transfer_learning.CTMTransferConfig] = None) -> udl_rating_framework.training.transfer_learning.CTMTransferLearningModel` [✓]

## udl_rating_framework.training.uncertainty_quantification

### Classes
- `CalibrationAnalyzer()` [✓]
- `DeepEnsembleUncertainty(models: List[udl_rating_framework.models.ctm_adapter.UDLRatingCTM])` [✓]
- `NeuronLevelUncertainty(analyze_memory: bool = True, neuron_diversity_threshold: float = 0.1)` [✓]
- `SynchronizationUncertainty(n_samples: int = 50, temporal_analysis: bool = True)` [✓]
- `UncertaintyAwarePredictor(model: udl_rating_framework.models.ctm_adapter.UDLRatingCTM, uncertainty_methods: Optional[List[udl_rating_framework.training.uncertainty_quantification.UncertaintyQuantifier]] = None, calibration_analyzer: Optional[udl_rating_framework.training.uncertainty_quantification.CalibrationAnalyzer] = None)` [✓]
- `UncertaintyEstimate(prediction: float, confidence: float, epistemic_uncertainty: float, aleatoric_uncertainty: float, total_uncertainty: float, confidence_interval_lower: float, confidence_interval_upper: float, confidence_level: float = 0.95, calibration_error: Optional[float] = None, reliability_score: Optional[float] = None, method: str = 'unknown', n_samples: int = 1) -> None` [✓]
- `UncertaintyQuantifier()` [✓]
- `VariationalInference(n_samples: int = 100)` [✓]

### Functions
- `bootstrap_confidence_intervals(predictions: numpy.ndarray, n_bootstrap: int = 1000, confidence_level: float = 0.95) -> Tuple[float, float]` [✓]
- `create_uncertainty_quantifier(method: str = 'monte_carlo_dropout', **kwargs) -> udl_rating_framework.training.uncertainty_quantification.UncertaintyQuantifier` [✓]

## udl_rating_framework.validation.api_validator

### Classes
- `APIDiscrepancy(element_name: str, discrepancy_type: str, documented_value: Optional[str] = None, actual_value: Optional[str] = None, severity: str = 'major', suggestion: Optional[str] = None) -> None` [✓]
- `APIElement(name: str, module: str, element_type: str, signature: Optional[str] = None, docstring: Optional[str] = None, parameters: List[str] = <factory>, return_type: Optional[str] = None, parent_class: Optional[str] = None, line_number: Optional[int] = None) -> None` [✓]
- `APIExtractor(package_name: str = 'udl_rating_framework')` [✓]
- `APIValidationReport(total_documented_apis: int = 0, total_actual_apis: int = 0, documented_apis: List[udl_rating_framework.validation.api_validator.APIElement] = <factory>, actual_apis: List[udl_rating_framework.validation.api_validator.APIElement] = <factory>, discrepancies: List[udl_rating_framework.validation.api_validator.APIDiscrepancy] = <factory>, undocumented_apis: List[udl_rating_framework.validation.api_validator.APIElement] = <factory>, orphaned_docs: List[str] = <factory>) -> None` [✓]
- `APIValidator(package_name: str = 'udl_rating_framework', docs_path: Optional[pathlib._local.Path] = None)` [✓]
- `DocumentedAPIExtractor(docs_path: pathlib._local.Path)` [✓]

### Functions
- `main()` [✓]

## udl_rating_framework.validation.link_validator

### Classes
- `Link(source_file: pathlib._local.Path, source_line: int, link_text: str, link_target: str, link_type: udl_rating_framework.validation.link_validator.LinkType) -> None` [✓]
- `LinkType(*values)` [✓]
- `LinkValidationReport(total_links: int = 0, valid_links: int = 0, broken_links: int = 0, results: List[udl_rating_framework.validation.link_validator.ValidationResult] = <factory>, broken_link_details: List[udl_rating_framework.validation.link_validator.ValidationResult] = <factory>) -> None` [✓]
- `LinkValidator(project_root: pathlib._local.Path)` [✓]
- `ValidationResult(link: udl_rating_framework.validation.link_validator.Link, is_valid: bool, message: str, resolved_path: Optional[pathlib._local.Path] = None) -> None` [✓]

### Functions
- `validate_project_links(project_root: str = '.') -> udl_rating_framework.validation.link_validator.LinkValidationReport` [✓]

## udl_rating_framework.visualization.activation_visualizer

### Classes
- `ActivationVisualizer(figsize: Tuple[int, int] = (12, 8))` [✓]

## udl_rating_framework.visualization.realtime_metrics

### Classes
- `MetricUpdate(timestamp: float, metric_name: str, value: float, udl_file: str, iteration: int) -> None` [✓]
- `RealTimeMetricsVisualizer(output_dir: str = 'realtime_visualizations')` [✓]

## udl_rating_framework.visualization.synchronization_visualizer

### Classes
- `SynchronizationVisualizer(figsize: Tuple[int, int] = (12, 8))` [✓]

## udl_rating_framework.visualization.web_visualizer

### Classes
- `WebVisualizer(output_dir: str = 'visualizations')` [✓]

## udl_rating_framework.visualization.webgl_visualizer

### Classes
- `WebGLVisualizer(output_dir: str = 'webgl_visualizations')` [✓]

