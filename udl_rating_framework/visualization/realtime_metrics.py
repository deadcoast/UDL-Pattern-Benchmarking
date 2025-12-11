"""
Real-time Metrics Visualization.

Provides real-time visualization of metric computation with live updates
and interactive dashboards for monitoring UDL analysis progress.
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class MetricUpdate:
    """Represents a single metric update."""
    timestamp: float
    metric_name: str
    value: float
    udl_file: str
    iteration: int


class RealTimeMetricsVisualizer:
    """
    Real-time metrics visualization system.
    
    Provides live updates of metric computations with:
    - WebSocket-based real-time updates
    - Interactive dashboards
    - Historical data tracking
    - Performance monitoring
    """
    
    def __init__(self, output_dir: str = "realtime_visualizations"):
        """Initialize real-time visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.metric_history: Dict[str, List[MetricUpdate]] = {}
        self.active_computations: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            "computation_time": [],
            "memory_usage": [],
            "throughput": []
        }
        
        # Real-time update callbacks
        self.update_callbacks: List[Callable[[MetricUpdate], None]] = []
        
        # Threading for real-time updates
        self.update_thread = None
        self.stop_updates = False
    
    def add_metric_update(self, 
                         metric_name: str, 
                         value: float, 
                         udl_file: str, 
                         iteration: int = 0):
        """Add a new metric update."""
        update = MetricUpdate(
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            udl_file=udl_file,
            iteration=iteration
        )
        
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append(update)
        
        # Trigger callbacks
        for callback in self.update_callbacks:
            callback(update)
    
    def start_computation_tracking(self, computation_id: str, udl_file: str):
        """Start tracking a computation."""
        self.active_computations[computation_id] = {
            "udl_file": udl_file,
            "start_time": time.time(),
            "metrics_computed": [],
            "status": "running"
        }
    
    def finish_computation_tracking(self, computation_id: str):
        """Finish tracking a computation."""
        if computation_id in self.active_computations:
            computation = self.active_computations[computation_id]
            computation["end_time"] = time.time()
            computation["status"] = "completed"
            computation["duration"] = computation["end_time"] - computation["start_time"]
            
            # Update performance metrics
            self.performance_metrics["computation_time"].append(computation["duration"])
    
    def create_realtime_dashboard(self, 
                                save_path: Optional[str] = None,
                                websocket_port: int = 8765) -> str:
        """
        Create real-time dashboard with WebSocket updates.
        
        Args:
            save_path: Optional path to save HTML file
            websocket_port: Port for WebSocket server
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "realtime_dashboard.html"
        else:
            save_path = Path(save_path)
        
        # Generate HTML with WebSocket integration
        html_content = self._generate_realtime_dashboard_html(websocket_port)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def create_performance_monitor(self, save_path: Optional[str] = None) -> str:
        """
        Create performance monitoring dashboard.
        
        Args:
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "performance_monitor.html"
        else:
            save_path = Path(save_path)
        
        # Generate performance monitoring HTML
        html_content = self._generate_performance_monitor_html()
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def create_metric_comparison_view(self, 
                                    metric_names: List[str],
                                    save_path: Optional[str] = None) -> str:
        """
        Create comparative view of multiple metrics.
        
        Args:
            metric_names: List of metrics to compare
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "metric_comparison.html"
        else:
            save_path = Path(save_path)
        
        # Prepare comparison data
        comparison_data = self._prepare_comparison_data(metric_names)
        
        # Generate comparison HTML
        html_content = self._generate_comparison_html(comparison_data)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def export_data(self, format: str = "json") -> str:
        """
        Export collected data in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'parquet')
            
        Returns:
            Path to exported file
        """
        timestamp = int(time.time())
        
        if format == "json":
            export_path = self.output_dir / f"metrics_export_{timestamp}.json"
            export_data = {
                "metric_history": {
                    name: [asdict(update) for update in updates]
                    for name, updates in self.metric_history.items()
                },
                "active_computations": self.active_computations,
                "performance_metrics": self.performance_metrics
            }
            
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)
        
        elif format == "csv":
            import pandas as pd
            
            export_path = self.output_dir / f"metrics_export_{timestamp}.csv"
            
            # Flatten metric history for CSV
            rows = []
            for metric_name, updates in self.metric_history.items():
                for update in updates:
                    rows.append({
                        "timestamp": update.timestamp,
                        "metric_name": update.metric_name,
                        "value": update.value,
                        "udl_file": update.udl_file,
                        "iteration": update.iteration
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(export_path, index=False)
        
        return str(export_path)
    
    def _prepare_comparison_data(self, metric_names: List[str]) -> Dict[str, Any]:
        """Prepare data for metric comparison."""
        comparison_data = {
            "metrics": metric_names,
            "data": {},
            "statistics": {}
        }
        
        for metric_name in metric_names:
            if metric_name in self.metric_history:
                updates = self.metric_history[metric_name]
                values = [update.value for update in updates]
                timestamps = [update.timestamp for update in updates]
                
                comparison_data["data"][metric_name] = {
                    "values": values,
                    "timestamps": timestamps,
                    "count": len(values)
                }
                
                if values:
                    comparison_data["statistics"][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "latest": values[-1]
                    }
        
        return comparison_data
    
    def _generate_realtime_dashboard_html(self, websocket_port: int) -> str:
        """Generate HTML for real-time dashboard."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Metrics Dashboard</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-connected {{ background: #4caf50; }}
        .status-disconnected {{ background: #f44336; }}
        .status-connecting {{ background: #ff9800; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2196f3;
            text-align: center;
            margin: 10px 0;
        }}
        
        .metric-name {{
            text-align: center;
            color: #666;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .metric-trend {{
            text-align: center;
            font-size: 0.9em;
        }}
        
        .trend-up {{ color: #4caf50; }}
        .trend-down {{ color: #f44336; }}
        .trend-stable {{ color: #666; }}
        
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .controls {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .control-group {{
            display: inline-block;
            margin: 0 15px;
        }}
        
        button {{
            background: #2196f3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
        }}
        
        button:hover {{ background: #1976d2; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        
        .log-container {{
            background: #1e1e1e;
            color: #00ff00;
            padding: 15px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            height: 200px;
            overflow-y: auto;
            margin-bottom: 20px;
        }}
        
        .log-entry {{
            margin: 2px 0;
            font-size: 0.9em;
        }}
        
        .log-timestamp {{
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>Real-time UDL Metrics Dashboard</h1>
        <div>
            <span class="status-indicator" id="connectionStatus"></span>
            <span id="connectionText">Connecting...</span>
            <span style="margin-left: 20px;">Updates: <span id="updateCount">0</span></span>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <button onclick="toggleUpdates()">Pause/Resume</button>
            <button onclick="clearData()">Clear Data</button>
            <button onclick="exportData()">Export Data</button>
        </div>
        <div class="control-group">
            <label>Update Rate: 
                <select id="updateRate" onchange="changeUpdateRate()">
                    <option value="100">10 Hz</option>
                    <option value="500" selected>2 Hz</option>
                    <option value="1000">1 Hz</option>
                    <option value="2000">0.5 Hz</option>
                </select>
            </label>
        </div>
        <div class="control-group">
            <label>History Length: 
                <select id="historyLength" onchange="changeHistoryLength()">
                    <option value="50">50 points</option>
                    <option value="100" selected>100 points</option>
                    <option value="200">200 points</option>
                    <option value="500">500 points</option>
                </select>
            </label>
        </div>
    </div>
    
    <div class="metrics-grid" id="metricsGrid">
        <!-- Metric cards will be dynamically generated -->
    </div>
    
    <div class="chart-container">
        <h3>Metric Evolution</h3>
        <div id="timeSeriesChart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>Metric Correlations</h3>
        <div id="correlationChart" style="height: 400px;"></div>
    </div>
    
    <div class="log-container" id="logContainer">
        <div class="log-entry">
            <span class="log-timestamp">[Starting]</span> Real-time dashboard initialized
        </div>
    </div>

    <script>
        // WebSocket connection
        let ws;
        let isConnected = false;
        let isPaused = false;
        let updateCount = 0;
        let updateRate = 500;
        let historyLength = 100;
        
        // Data storage
        let metricsData = {{}};
        let timeSeriesData = [];
        
        // UI elements
        const connectionStatus = document.getElementById('connectionStatus');
        const connectionText = document.getElementById('connectionText');
        const updateCountElement = document.getElementById('updateCount');
        const metricsGrid = document.getElementById('metricsGrid');
        const logContainer = document.getElementById('logContainer');
        
        function connectWebSocket() {{
            try {{
                ws = new WebSocket(`ws://localhost:{websocket_port}`);
                
                ws.onopen = function(event) {{
                    isConnected = true;
                    updateConnectionStatus();
                    addLogEntry('Connected to metrics server');
                }};
                
                ws.onmessage = function(event) {{
                    if (!isPaused) {{
                        const data = JSON.parse(event.data);
                        handleMetricUpdate(data);
                    }}
                }};
                
                ws.onclose = function(event) {{
                    isConnected = false;
                    updateConnectionStatus();
                    addLogEntry('Disconnected from metrics server');
                    
                    // Attempt to reconnect after 3 seconds
                    setTimeout(connectWebSocket, 3000);
                }};
                
                ws.onerror = function(error) {{
                    addLogEntry('WebSocket error: ' + error);
                }};
                
            }} catch (error) {{
                addLogEntry('Failed to connect: ' + error);
                setTimeout(connectWebSocket, 3000);
            }}
        }}
        
        function updateConnectionStatus() {{
            if (isConnected) {{
                connectionStatus.className = 'status-indicator status-connected';
                connectionText.textContent = 'Connected';
            }} else {{
                connectionStatus.className = 'status-indicator status-disconnected';
                connectionText.textContent = 'Disconnected';
            }}
        }}
        
        function handleMetricUpdate(data) {{
            updateCount++;
            updateCountElement.textContent = updateCount;
            
            const metricName = data.metric_name;
            const value = data.value;
            const timestamp = data.timestamp;
            
            // Update metrics data
            if (!metricsData[metricName]) {{
                metricsData[metricName] = {{
                    values: [],
                    timestamps: [],
                    latest: 0,
                    trend: 'stable'
                }};
            }}
            
            const metric = metricsData[metricName];
            metric.values.push(value);
            metric.timestamps.push(timestamp);
            metric.latest = value;
            
            // Keep only recent history
            if (metric.values.length > historyLength) {{
                metric.values.shift();
                metric.timestamps.shift();
            }}
            
            // Calculate trend
            if (metric.values.length >= 2) {{
                const recent = metric.values.slice(-5);
                const older = metric.values.slice(-10, -5);
                if (recent.length > 0 && older.length > 0) {{
                    const recentAvg = recent.reduce((a, b) => a + b) / recent.length;
                    const olderAvg = older.reduce((a, b) => a + b) / older.length;
                    const diff = recentAvg - olderAvg;
                    
                    if (Math.abs(diff) < 0.01) {{
                        metric.trend = 'stable';
                    }} else if (diff > 0) {{
                        metric.trend = 'up';
                    }} else {{
                        metric.trend = 'down';
                    }}
                }}
            }}
            
            // Update UI
            updateMetricCards();
            updateCharts();
            
            addLogEntry(`${{metricName}}: ${{value.toFixed(4)}} (${{data.udl_file}})`);
        }}
        
        function updateMetricCards() {{
            metricsGrid.innerHTML = '';
            
            Object.keys(metricsData).forEach(metricName => {{
                const metric = metricsData[metricName];
                const card = document.createElement('div');
                card.className = 'metric-card';
                
                const trendClass = `trend-${{metric.trend}}`;
                const trendSymbol = metric.trend === 'up' ? '↗' : 
                                  metric.trend === 'down' ? '↘' : '→';
                
                card.innerHTML = `
                    <div class="metric-name">${{metricName}}</div>
                    <div class="metric-value">${{metric.latest.toFixed(4)}}</div>
                    <div class="metric-trend ${{trendClass}}">
                        ${{trendSymbol}} ${{metric.trend.toUpperCase()}}
                    </div>
                `;
                
                metricsGrid.appendChild(card);
            }});
        }}
        
        function updateCharts() {{
            updateTimeSeriesChart();
            updateCorrelationChart();
        }}
        
        function updateTimeSeriesChart() {{
            const traces = [];
            
            Object.keys(metricsData).forEach(metricName => {{
                const metric = metricsData[metricName];
                traces.push({{
                    x: metric.timestamps.map(t => new Date(t * 1000)),
                    y: metric.values,
                    name: metricName,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{ width: 2 }}
                }});
            }});
            
            const layout = {{
                title: 'Metric Values Over Time',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Value', range: [0, 1] }},
                showlegend: true,
                margin: {{ t: 50, r: 50, b: 50, l: 50 }}
            }};
            
            Plotly.newPlot('timeSeriesChart', traces, layout, {{ responsive: true }});
        }}
        
        function updateCorrelationChart() {{
            const metricNames = Object.keys(metricsData);
            if (metricNames.length < 2) return;
            
            // Calculate correlation matrix
            const correlations = [];
            const labels = [];
            
            for (let i = 0; i < metricNames.length; i++) {{
                correlations[i] = [];
                labels[i] = metricNames[i];
                
                for (let j = 0; j < metricNames.length; j++) {{
                    const corr = calculateCorrelation(
                        metricsData[metricNames[i]].values,
                        metricsData[metricNames[j]].values
                    );
                    correlations[i][j] = corr;
                }}
            }}
            
            const trace = {{
                z: correlations,
                x: labels,
                y: labels,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmid: 0,
                zmin: -1,
                zmax: 1
            }};
            
            const layout = {{
                title: 'Metric Correlations',
                xaxis: {{ title: 'Metrics' }},
                yaxis: {{ title: 'Metrics' }},
                margin: {{ t: 50, r: 50, b: 100, l: 100 }}
            }};
            
            Plotly.newPlot('correlationChart', [trace], layout, {{ responsive: true }});
        }}
        
        function calculateCorrelation(x, y) {{
            const n = Math.min(x.length, y.length);
            if (n < 2) return 0;
            
            const meanX = x.slice(0, n).reduce((a, b) => a + b) / n;
            const meanY = y.slice(0, n).reduce((a, b) => a + b) / n;
            
            let numerator = 0;
            let sumXX = 0;
            let sumYY = 0;
            
            for (let i = 0; i < n; i++) {{
                const dx = x[i] - meanX;
                const dy = y[i] - meanY;
                numerator += dx * dy;
                sumXX += dx * dx;
                sumYY += dy * dy;
            }}
            
            const denominator = Math.sqrt(sumXX * sumYY);
            return denominator === 0 ? 0 : numerator / denominator;
        }}
        
        function addLogEntry(message) {{
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-timestamp">[${{timestamp}}]</span> ${{message}}`;
            
            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Keep only last 100 log entries
            while (logContainer.children.length > 100) {{
                logContainer.removeChild(logContainer.firstChild);
            }}
        }}
        
        function toggleUpdates() {{
            isPaused = !isPaused;
            addLogEntry(isPaused ? 'Updates paused' : 'Updates resumed');
        }}
        
        function clearData() {{
            metricsData = {{}};
            updateCount = 0;
            updateCountElement.textContent = '0';
            metricsGrid.innerHTML = '';
            updateCharts();
            addLogEntry('Data cleared');
        }}
        
        function exportData() {{
            const dataStr = JSON.stringify(metricsData, null, 2);
            const dataBlob = new Blob([dataStr], {{ type: 'application/json' }});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `metrics_data_${{Date.now()}}.json`;
            link.click();
            
            URL.revokeObjectURL(url);
            addLogEntry('Data exported');
        }}
        
        function changeUpdateRate() {{
            updateRate = parseInt(document.getElementById('updateRate').value);
            addLogEntry(`Update rate changed to ${{1000/updateRate}} Hz`);
        }}
        
        function changeHistoryLength() {{
            historyLength = parseInt(document.getElementById('historyLength').value);
            addLogEntry(`History length changed to ${{historyLength}} points`);
        }}
        
        // Initialize
        connectWebSocket();
        updateConnectionStatus();
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', function() {{
            if (document.hidden) {{
                addLogEntry('Dashboard hidden - pausing updates');
                isPaused = true;
            }} else {{
                addLogEntry('Dashboard visible - resuming updates');
                isPaused = false;
            }}
        }});
    </script>
</body>
</html>
        """
    
    def _generate_performance_monitor_html(self) -> str:
        """Generate HTML for performance monitoring dashboard."""
        performance_data = json.dumps(self.performance_metrics)
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Monitor</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        .performance-header {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #ff6b6b;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }}
        
        .alert-danger {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }}
        
        .alert-success {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }}
    </style>
</head>
<body>
    <div class="performance-header">
        <h1>UDL Rating Framework - Performance Monitor</h1>
        <p>Real-time performance metrics and system health monitoring</p>
    </div>
    
    <div class="stats-grid" id="statsGrid">
        <!-- Performance stats will be dynamically generated -->
    </div>
    
    <div id="alerts">
        <!-- Performance alerts will appear here -->
    </div>
    
    <div class="chart-container">
        <h3>Computation Time Distribution</h3>
        <div id="computationTimeChart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>Memory Usage Over Time</h3>
        <div id="memoryUsageChart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>Throughput Analysis</h3>
        <div id="throughputChart" style="height: 400px;"></div>
    </div>

    <script>
        const performanceData = {performance_data};
        
        function updatePerformanceStats() {{
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = '';
            
            // Computation time stats
            if (performanceData.computation_time.length > 0) {{
                const times = performanceData.computation_time;
                const avgTime = times.reduce((a, b) => a + b) / times.length;
                const maxTime = Math.max(...times);
                const minTime = Math.min(...times);
                
                statsGrid.innerHTML += `
                    <div class="stat-card">
                        <div class="stat-value">${{avgTime.toFixed(2)}}s</div>
                        <div class="stat-label">Average Computation Time</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{maxTime.toFixed(2)}}s</div>
                        <div class="stat-label">Max Computation Time</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{minTime.toFixed(2)}}s</div>
                        <div class="stat-label">Min Computation Time</div>
                    </div>
                `;
            }}
            
            // Memory usage stats
            if (performanceData.memory_usage.length > 0) {{
                const memory = performanceData.memory_usage;
                const avgMemory = memory.reduce((a, b) => a + b) / memory.length;
                const maxMemory = Math.max(...memory);
                
                statsGrid.innerHTML += `
                    <div class="stat-card">
                        <div class="stat-value">${{(avgMemory / 1024 / 1024).toFixed(1)}}MB</div>
                        <div class="stat-label">Average Memory Usage</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{(maxMemory / 1024 / 1024).toFixed(1)}}MB</div>
                        <div class="stat-label">Peak Memory Usage</div>
                    </div>
                `;
            }}
            
            // Throughput stats
            if (performanceData.throughput.length > 0) {{
                const throughput = performanceData.throughput;
                const avgThroughput = throughput.reduce((a, b) => a + b) / throughput.length;
                
                statsGrid.innerHTML += `
                    <div class="stat-card">
                        <div class="stat-value">${{avgThroughput.toFixed(1)}}</div>
                        <div class="stat-label">Files/Second</div>
                    </div>
                `;
            }}
        }}
        
        function updatePerformanceAlerts() {{
            const alertsContainer = document.getElementById('alerts');
            alertsContainer.innerHTML = '';
            
            // Check for performance issues
            if (performanceData.computation_time.length > 0) {{
                const avgTime = performanceData.computation_time.reduce((a, b) => a + b) / performanceData.computation_time.length;
                
                if (avgTime > 10) {{
                    alertsContainer.innerHTML += `
                        <div class="alert alert-danger">
                            <strong>Performance Warning:</strong> Average computation time is high (${{avgTime.toFixed(2)}}s). Consider optimizing metric calculations.
                        </div>
                    `;
                }} else if (avgTime > 5) {{
                    alertsContainer.innerHTML += `
                        <div class="alert alert-warning">
                            <strong>Performance Notice:</strong> Computation time is elevated (${{avgTime.toFixed(2)}}s). Monitor for trends.
                        </div>
                    `;
                }} else {{
                    alertsContainer.innerHTML += `
                        <div class="alert alert-success">
                            <strong>Performance Good:</strong> Computation times are within acceptable range (${{avgTime.toFixed(2)}}s).
                        </div>
                    `;
                }}
            }}
            
            if (performanceData.memory_usage.length > 0) {{
                const maxMemory = Math.max(...performanceData.memory_usage);
                const maxMemoryMB = maxMemory / 1024 / 1024;
                
                if (maxMemoryMB > 1000) {{
                    alertsContainer.innerHTML += `
                        <div class="alert alert-danger">
                            <strong>Memory Warning:</strong> Peak memory usage is high (${{maxMemoryMB.toFixed(1)}}MB). Check for memory leaks.
                        </div>
                    `;
                }} else if (maxMemoryMB > 500) {{
                    alertsContainer.innerHTML += `
                        <div class="alert alert-warning">
                            <strong>Memory Notice:</strong> Memory usage is elevated (${{maxMemoryMB.toFixed(1)}}MB). Monitor trends.
                        </div>
                    `;
                }}
            }}
        }}
        
        function updateCharts() {{
            updateComputationTimeChart();
            updateMemoryUsageChart();
            updateThroughputChart();
        }}
        
        function updateComputationTimeChart() {{
            if (performanceData.computation_time.length === 0) return;
            
            const trace = {{
                x: performanceData.computation_time,
                type: 'histogram',
                nbinsx: 20,
                name: 'Computation Time',
                marker: {{ color: '#ff6b6b' }}
            }};
            
            const layout = {{
                title: 'Distribution of Computation Times',
                xaxis: {{ title: 'Time (seconds)' }},
                yaxis: {{ title: 'Frequency' }},
                margin: {{ t: 50, r: 50, b: 50, l: 50 }}
            }};
            
            Plotly.newPlot('computationTimeChart', [trace], layout, {{ responsive: true }});
        }}
        
        function updateMemoryUsageChart() {{
            if (performanceData.memory_usage.length === 0) return;
            
            const trace = {{
                y: performanceData.memory_usage.map(m => m / 1024 / 1024), // Convert to MB
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Memory Usage',
                line: {{ color: '#ff6b6b', width: 2 }}
            }};
            
            const layout = {{
                title: 'Memory Usage Over Time',
                xaxis: {{ title: 'Sample' }},
                yaxis: {{ title: 'Memory (MB)' }},
                margin: {{ t: 50, r: 50, b: 50, l: 50 }}
            }};
            
            Plotly.newPlot('memoryUsageChart', [trace], layout, {{ responsive: true }});
        }}
        
        function updateThroughputChart() {{
            if (performanceData.throughput.length === 0) return;
            
            const trace = {{
                y: performanceData.throughput,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Throughput',
                line: {{ color: '#ff6b6b', width: 2 }}
            }};
            
            const layout = {{
                title: 'Processing Throughput Over Time',
                xaxis: {{ title: 'Sample' }},
                yaxis: {{ title: 'Files/Second' }},
                margin: {{ t: 50, r: 50, b: 50, l: 50 }}
            }};
            
            Plotly.newPlot('throughputChart', [trace], layout, {{ responsive: true }});
        }}
        
        // Initialize
        updatePerformanceStats();
        updatePerformanceAlerts();
        updateCharts();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {{
            updatePerformanceStats();
            updatePerformanceAlerts();
            updateCharts();
        }}, 30000);
    </script>
</body>
</html>
        """
    
    def _generate_comparison_html(self, comparison_data: Dict[str, Any]) -> str:
        """Generate HTML for metric comparison view."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Metric Comparison View</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        .comparison-header {{
            background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .metrics-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .metric-summary-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .metric-name {{
            font-weight: bold;
            color: #4ecdc4;
            margin-bottom: 10px;
        }}
        
        .metric-stats {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .controls {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .control-group {{
            display: inline-block;
            margin: 0 15px;
        }}
        
        button {{
            background: #4ecdc4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
        }}
        
        button:hover {{ background: #44a08d; }}
    </style>
</head>
<body>
    <div class="comparison-header">
        <h1>Metric Comparison Analysis</h1>
        <p>Comparative analysis of {len(comparison_data['metrics'])} metrics</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <button onclick="showAllMetrics()">Show All</button>
            <button onclick="showTopPerformers()">Top Performers</button>
            <button onclick="showBottomPerformers()">Bottom Performers</button>
        </div>
        <div class="control-group">
            <label>Chart Type: 
                <select id="chartType" onchange="updateChartType()">
                    <option value="line">Line Chart</option>
                    <option value="box">Box Plot</option>
                    <option value="violin">Violin Plot</option>
                    <option value="scatter">Scatter Plot</option>
                </select>
            </label>
        </div>
    </div>
    
    <div class="metrics-summary" id="metricsSummary">
        <!-- Metric summary cards will be generated here -->
    </div>
    
    <div class="chart-container">
        <h3>Metric Comparison</h3>
        <div id="comparisonChart" style="height: 500px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>Statistical Summary</h3>
        <div id="statisticsChart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>Metric Relationships</h3>
        <div id="relationshipChart" style="height: 400px;"></div>
    </div>

    <script>
        const comparisonData = {json.dumps(comparison_data)};
        let currentChartType = 'line';
        
        function initializeComparison() {{
            updateMetricsSummary();
            updateComparisonChart();
            updateStatisticsChart();
            updateRelationshipChart();
        }}
        
        function updateMetricsSummary() {{
            const summaryContainer = document.getElementById('metricsSummary');
            summaryContainer.innerHTML = '';
            
            comparisonData.metrics.forEach(metricName => {{
                if (comparisonData.statistics[metricName]) {{
                    const stats = comparisonData.statistics[metricName];
                    const card = document.createElement('div');
                    card.className = 'metric-summary-card';
                    
                    card.innerHTML = `
                        <div class="metric-name">${{metricName}}</div>
                        <div class="metric-stats">
                            <div>Mean: ${{stats.mean.toFixed(4)}}</div>
                            <div>Std: ${{stats.std.toFixed(4)}}</div>
                            <div>Range: ${{stats.min.toFixed(4)}} - ${{stats.max.toFixed(4)}}</div>
                            <div>Latest: ${{stats.latest.toFixed(4)}}</div>
                            <div>Count: ${{comparisonData.data[metricName].count}}</div>
                        </div>
                    `;
                    
                    summaryContainer.appendChild(card);
                }}
            }});
        }}
        
        function updateComparisonChart() {{
            const traces = [];
            
            comparisonData.metrics.forEach(metricName => {{
                if (comparisonData.data[metricName]) {{
                    const data = comparisonData.data[metricName];
                    
                    if (currentChartType === 'line') {{
                        traces.push({{
                            x: data.timestamps.map(t => new Date(t * 1000)),
                            y: data.values,
                            name: metricName,
                            type: 'scatter',
                            mode: 'lines+markers'
                        }});
                    }} else if (currentChartType === 'box') {{
                        traces.push({{
                            y: data.values,
                            name: metricName,
                            type: 'box'
                        }});
                    }} else if (currentChartType === 'violin') {{
                        traces.push({{
                            y: data.values,
                            name: metricName,
                            type: 'violin'
                        }});
                    }}
                }}
            }});
            
            const layout = {{
                title: `Metric Comparison (${{currentChartType.charAt(0).toUpperCase() + currentChartType.slice(1)}})`,
                xaxis: {{ title: currentChartType === 'line' ? 'Time' : 'Metric' }},
                yaxis: {{ title: 'Value', range: [0, 1] }},
                showlegend: true,
                margin: {{ t: 50, r: 50, b: 50, l: 50 }}
            }};
            
            Plotly.newPlot('comparisonChart', traces, layout, {{ responsive: true }});
        }}
        
        function updateStatisticsChart() {{
            const metrics = comparisonData.metrics.filter(m => comparisonData.statistics[m]);
            const means = metrics.map(m => comparisonData.statistics[m].mean);
            const stds = metrics.map(m => comparisonData.statistics[m].std);
            
            const trace1 = {{
                x: metrics,
                y: means,
                name: 'Mean',
                type: 'bar',
                marker: {{ color: '#4ecdc4' }}
            }};
            
            const trace2 = {{
                x: metrics,
                y: stds,
                name: 'Standard Deviation',
                type: 'bar',
                yaxis: 'y2',
                marker: {{ color: '#ff6b6b' }}
            }};
            
            const layout = {{
                title: 'Statistical Summary',
                xaxis: {{ title: 'Metrics' }},
                yaxis: {{ title: 'Mean Value', side: 'left' }},
                yaxis2: {{ 
                    title: 'Standard Deviation', 
                    side: 'right', 
                    overlaying: 'y' 
                }},
                margin: {{ t: 50, r: 50, b: 100, l: 50 }}
            }};
            
            Plotly.newPlot('statisticsChart', [trace1, trace2], layout, {{ responsive: true }});
        }}
        
        function updateRelationshipChart() {{
            if (comparisonData.metrics.length < 2) return;
            
            const metric1 = comparisonData.metrics[0];
            const metric2 = comparisonData.metrics[1];
            
            if (comparisonData.data[metric1] && comparisonData.data[metric2]) {{
                const data1 = comparisonData.data[metric1].values;
                const data2 = comparisonData.data[metric2].values;
                const minLength = Math.min(data1.length, data2.length);
                
                const trace = {{
                    x: data1.slice(0, minLength),
                    y: data2.slice(0, minLength),
                    mode: 'markers',
                    type: 'scatter',
                    name: `${{metric1}} vs ${{metric2}}`,
                    marker: {{ 
                        color: '#4ecdc4',
                        size: 8,
                        opacity: 0.7
                    }}
                }};
                
                const layout = {{
                    title: `Relationship: ${{metric1}} vs ${{metric2}}`,
                    xaxis: {{ title: metric1 }},
                    yaxis: {{ title: metric2 }},
                    margin: {{ t: 50, r: 50, b: 50, l: 50 }}
                }};
                
                Plotly.newPlot('relationshipChart', [trace], layout, {{ responsive: true }});
            }}
        }}
        
        function updateChartType() {{
            currentChartType = document.getElementById('chartType').value;
            updateComparisonChart();
        }}
        
        function showAllMetrics() {{
            // Show all metrics (default behavior)
            updateComparisonChart();
        }}
        
        function showTopPerformers() {{
            // Filter to show only top performing metrics
            const sortedMetrics = comparisonData.metrics
                .filter(m => comparisonData.statistics[m])
                .sort((a, b) => comparisonData.statistics[b].mean - comparisonData.statistics[a].mean)
                .slice(0, 3);
            
            // Update chart with filtered data
            // Implementation would filter the traces
            updateComparisonChart();
        }}
        
        function showBottomPerformers() {{
            // Filter to show only bottom performing metrics
            const sortedMetrics = comparisonData.metrics
                .filter(m => comparisonData.statistics[m])
                .sort((a, b) => comparisonData.statistics[a].mean - comparisonData.statistics[b].mean)
                .slice(0, 3);
            
            // Update chart with filtered data
            updateComparisonChart();
        }}
        
        // Initialize
        initializeComparison();
    </script>
</body>
</html>
        """