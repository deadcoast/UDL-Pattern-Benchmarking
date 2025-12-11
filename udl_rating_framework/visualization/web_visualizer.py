"""
Web-based Interactive Visualization Utilities.

Provides tools for creating interactive web-based visualizations using D3.js
and other web technologies for enhanced UDL analysis.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import networkx as nx
from pathlib import Path
from ..models.ctm_adapter import TrackingData
from ..core.representation import UDLRepresentation


class WebVisualizer:
    """
    Creates interactive web-based visualizations for UDL analysis.
    
    Generates HTML pages with embedded D3.js visualizations for:
    - Interactive grammar graphs
    - Real-time metric computation
    - CTM processing animations
    - Grammar dependency flows
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize web visualizer.
        
        Args:
            output_dir: Directory to save HTML visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Copy static assets (D3.js, CSS, etc.)
        self._setup_static_assets()
    
    def _setup_static_assets(self):
        """Set up static assets for web visualizations."""
        static_dir = self.output_dir / "static"
        static_dir.mkdir(exist_ok=True)
        
        # Create CSS file for styling
        css_content = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .visualization-panel {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .controls {
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .metric-display {
            display: inline-block;
            margin: 5px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 5px;
            min-width: 120px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #1976d2;
        }
        
        .node {
            stroke: #333;
            stroke-width: 1.5px;
        }
        
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        
        .grammar-node {
            cursor: pointer;
        }
        
        .grammar-node:hover {
            stroke: #ff5722;
            stroke-width: 3px;
        }
        
        .tooltip {
            position: absolute;
            text-align: center;
            padding: 8px;
            font: 12px sans-serif;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
        }
        
        .animation-controls {
            text-align: center;
            margin: 20px 0;
        }
        
        .animation-controls button {
            margin: 0 5px;
            padding: 8px 16px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .animation-controls button:hover {
            background: #1976d2;
        }
        
        .animation-controls button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.3s ease;
        }
        """
        
        with open(static_dir / "style.css", "w") as f:
            f.write(css_content)
    
    def create_interactive_grammar_graph(self, 
                                       udl: UDLRepresentation,
                                       save_path: Optional[str] = None) -> str:
        """
        Create interactive 3D grammar graph visualization.
        
        Args:
            udl: UDL representation with grammar graph
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "grammar_graph.html"
        else:
            save_path = Path(save_path)
        
        # Extract graph data
        graph = udl.get_grammar_graph()
        
        # Convert NetworkX graph to D3.js format
        nodes = []
        links = []
        
        # Create nodes
        for node_id, node_data in graph.nodes(data=True):
            nodes.append({
                "id": str(node_id),
                "label": str(node_id),
                "type": node_data.get("type", "unknown"),
                "size": len(str(node_id)) * 2 + 10,
                "color": self._get_node_color(node_data.get("type", "unknown"))
            })
        
        # Create links
        for source, target, edge_data in graph.edges(data=True):
            links.append({
                "source": str(source),
                "target": str(target),
                "weight": edge_data.get("weight", 1),
                "type": edge_data.get("type", "production")
            })
        
        # Generate HTML with embedded D3.js visualization
        html_content = self._generate_grammar_graph_html(nodes, links)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def create_ctm_animation(self, 
                           tracking_data: TrackingData,
                           save_path: Optional[str] = None) -> str:
        """
        Create animated visualization of CTM processing over time.
        
        Args:
            tracking_data: TrackingData with recorded activations and synchronization
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "ctm_animation.html"
        else:
            save_path = Path(save_path)
        
        # Prepare animation data
        animation_data = self._prepare_ctm_animation_data(tracking_data)
        
        # Generate HTML with animation
        html_content = self._generate_ctm_animation_html(animation_data)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def create_real_time_metrics_dashboard(self, 
                                         metric_history: Dict[str, List[float]],
                                         save_path: Optional[str] = None) -> str:
        """
        Create real-time metrics computation visualization.
        
        Args:
            metric_history: Dictionary mapping metric names to value histories
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "metrics_dashboard.html"
        else:
            save_path = Path(save_path)
        
        # Generate HTML with real-time dashboard
        html_content = self._generate_metrics_dashboard_html(metric_history)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def create_dependency_flow_diagram(self, 
                                     udl: UDLRepresentation,
                                     save_path: Optional[str] = None) -> str:
        """
        Create grammar dependency flow diagram.
        
        Args:
            udl: UDL representation with grammar dependencies
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "dependency_flow.html"
        else:
            save_path = Path(save_path)
        
        # Extract dependency information
        dependencies = self._extract_dependencies(udl)
        
        # Generate HTML with flow diagram
        html_content = self._generate_dependency_flow_html(dependencies)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node based on type."""
        color_map = {
            "terminal": "#4caf50",
            "non_terminal": "#2196f3",
            "rule": "#ff9800",
            "constraint": "#f44336",
            "unknown": "#9e9e9e"
        }
        return color_map.get(node_type, "#9e9e9e")
    
    def _prepare_ctm_animation_data(self, tracking_data: TrackingData) -> Dict[str, Any]:
        """Prepare data for CTM animation."""
        return {
            "iterations": tracking_data.iterations,
            "n_neurons": tracking_data.n_neurons,
            "activations": tracking_data.post_activations.tolist(),
            "synchronization": tracking_data.synch_out.tolist(),
            "attention_weights": tracking_data.attention_weights.tolist() if tracking_data.attention_weights is not None else None
        }
    
    def _extract_dependencies(self, udl: UDLRepresentation) -> Dict[str, Any]:
        """Extract dependency information from UDL."""
        graph = udl.get_grammar_graph()
        
        # Compute strongly connected components
        sccs = list(nx.strongly_connected_components(graph))
        
        # Compute topological ordering (if DAG)
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            topo_order = list(graph.nodes())
        
        return {
            "nodes": [{"id": str(node), "group": i} for i, scc in enumerate(sccs) for node in scc],
            "links": [{"source": str(u), "target": str(v)} for u, v in graph.edges()],
            "sccs": [list(map(str, scc)) for scc in sccs],
            "topo_order": list(map(str, topo_order))
        }
    
    def _generate_grammar_graph_html(self, nodes: List[Dict], links: List[Dict]) -> str:
        """Generate HTML for interactive grammar graph."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Grammar Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/three@0.147.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.147.0/examples/js/controls/OrbitControls.js"></script>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div class="container">
        <h1>Interactive Grammar Graph Visualization</h1>
        
        <div class="controls">
            <label>
                <input type="checkbox" id="show3d" onchange="toggle3D()"> 3D View
            </label>
            <label>
                Node Size: <input type="range" id="nodeSize" min="5" max="50" value="20" onchange="updateNodeSize()">
            </label>
            <label>
                Link Strength: <input type="range" id="linkStrength" min="0.1" max="2" step="0.1" value="1" onchange="updateLinkStrength()">
            </label>
        </div>
        
        <div id="graph-container" class="visualization-panel">
            <svg id="graph-svg" width="800" height="600"></svg>
            <div id="graph-3d" style="display: none;"></div>
        </div>
        
        <div class="tooltip"></div>
    </div>

    <script>
        const nodes = {json.dumps(nodes)};
        const links = {json.dumps(links)};
        
        let simulation, svg, node, link, tooltip;
        let is3D = false;
        let scene, camera, renderer, controls;
        
        function initializeVisualization() {{
            svg = d3.select("#graph-svg");
            tooltip = d3.select(".tooltip");
            
            // Initialize 2D visualization
            initialize2D();
        }}
        
        function initialize2D() {{
            svg.selectAll("*").remove();
            
            const width = 800;
            const height = 600;
            
            simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).strength(1))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2));
            
            link = svg.append("g")
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.sqrt(d.weight));
            
            node = svg.append("g")
                .selectAll("circle")
                .data(nodes)
                .enter().append("circle")
                .attr("class", "grammar-node")
                .attr("r", d => d.size)
                .attr("fill", d => d.color)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended))
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);
            
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
            }});
        }}
        
        function initialize3D() {{
            const container = document.getElementById('graph-3d');
            container.innerHTML = '';
            
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(75, 800/600, 0.1, 1000);
            renderer = new THREE.WebGLRenderer();
            renderer.setSize(800, 600);
            container.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            
            // Add nodes as spheres
            nodes.forEach(node => {{
                const geometry = new THREE.SphereGeometry(node.size / 10, 32, 32);
                const material = new THREE.MeshBasicMaterial({{ color: node.color }});
                const sphere = new THREE.Mesh(geometry, material);
                
                sphere.position.set(
                    (Math.random() - 0.5) * 100,
                    (Math.random() - 0.5) * 100,
                    (Math.random() - 0.5) * 100
                );
                
                scene.add(sphere);
            }});
            
            // Add links as lines
            links.forEach(link => {{
                const material = new THREE.LineBasicMaterial({{ color: 0x999999 }});
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, 0),
                    new THREE.Vector3(10, 10, 10)
                ]);
                const line = new THREE.Line(geometry, material);
                scene.add(line);
            }});
            
            camera.position.z = 50;
            
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }}
        
        function toggle3D() {{
            is3D = !is3D;
            if (is3D) {{
                document.getElementById('graph-svg').style.display = 'none';
                document.getElementById('graph-3d').style.display = 'block';
                initialize3D();
            }} else {{
                document.getElementById('graph-svg').style.display = 'block';
                document.getElementById('graph-3d').style.display = 'none';
                initialize2D();
            }}
        }}
        
        function updateNodeSize() {{
            const size = document.getElementById('nodeSize').value;
            if (!is3D) {{
                node.attr("r", d => d.size * size / 20);
            }}
        }}
        
        function updateLinkStrength() {{
            const strength = document.getElementById('linkStrength').value;
            if (!is3D) {{
                simulation.force("link").strength(strength);
                simulation.alpha(1).restart();
            }}
        }}
        
        function showTooltip(event, d) {{
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`Node: ${{d.label}}<br/>Type: ${{d.type}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }}
        
        function hideTooltip() {{
            tooltip.transition().duration(500).style("opacity", 0);
        }}
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Initialize on load
        initializeVisualization();
    </script>
</body>
</html>
        """
    
    def _generate_ctm_animation_html(self, animation_data: Dict[str, Any]) -> str:
        """Generate HTML for CTM processing animation."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>CTM Processing Animation</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div class="container">
        <h1>CTM Processing Animation</h1>
        
        <div class="animation-controls">
            <button onclick="playAnimation()">Play</button>
            <button onclick="pauseAnimation()">Pause</button>
            <button onclick="resetAnimation()">Reset</button>
            <label>
                Speed: <input type="range" id="speed" min="1" max="10" value="5" onchange="updateSpeed()">
            </label>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress"></div>
        </div>
        
        <div class="visualization-panel">
            <h3>Neuron Activations</h3>
            <svg id="activations-svg" width="800" height="300"></svg>
        </div>
        
        <div class="visualization-panel">
            <h3>Synchronization Evolution</h3>
            <svg id="synchronization-svg" width="800" height="300"></svg>
        </div>
        
        <div class="visualization-panel">
            <h3>Current Iteration: <span id="current-iteration">0</span></h3>
        </div>
    </div>

    <script>
        const animationData = {json.dumps(animation_data)};
        
        let currentIteration = 0;
        let isPlaying = false;
        let animationSpeed = 500; // milliseconds
        let animationInterval;
        
        const activationsSvg = d3.select("#activations-svg");
        const synchronizationSvg = d3.select("#synchronization-svg");
        
        function initializeAnimation() {{
            setupActivationsVisualization();
            setupSynchronizationVisualization();
            updateVisualization();
        }}
        
        function setupActivationsVisualization() {{
            const width = 800;
            const height = 300;
            const margin = {{ top: 20, right: 20, bottom: 30, left: 40 }};
            
            activationsSvg.selectAll("*").remove();
            
            const xScale = d3.scaleLinear()
                .domain([0, animationData.n_neurons - 1])
                .range([margin.left, width - margin.right]);
            
            const yScale = d3.scaleLinear()
                .domain([-1, 1])
                .range([height - margin.bottom, margin.top]);
            
            // Add axes
            activationsSvg.append("g")
                .attr("transform", `translate(0,${{height - margin.bottom}})`)
                .call(d3.axisBottom(xScale));
            
            activationsSvg.append("g")
                .attr("transform", `translate(${{margin.left}},0)`)
                .call(d3.axisLeft(yScale));
            
            // Add neuron bars
            activationsSvg.selectAll(".neuron-bar")
                .data(d3.range(animationData.n_neurons))
                .enter().append("rect")
                .attr("class", "neuron-bar")
                .attr("x", d => xScale(d) - 2)
                .attr("width", 4)
                .attr("y", yScale(0))
                .attr("height", 0)
                .attr("fill", "#2196f3");
        }}
        
        function setupSynchronizationVisualization() {{
            const width = 800;
            const height = 300;
            const margin = {{ top: 20, right: 20, bottom: 30, left: 40 }};
            
            synchronizationSvg.selectAll("*").remove();
            
            const synchData = animationData.synchronization[0][0]; // First batch, first iteration
            const xScale = d3.scaleLinear()
                .domain([0, synchData.length - 1])
                .range([margin.left, width - margin.right]);
            
            const yScale = d3.scaleLinear()
                .domain([0, 1])
                .range([height - margin.bottom, margin.top]);
            
            // Add axes
            synchronizationSvg.append("g")
                .attr("transform", `translate(0,${{height - margin.bottom}})`)
                .call(d3.axisBottom(xScale));
            
            synchronizationSvg.append("g")
                .attr("transform", `translate(${{margin.left}},0)`)
                .call(d3.axisLeft(yScale));
            
            // Add synchronization line
            const line = d3.line()
                .x((d, i) => xScale(i))
                .y(d => yScale(d));
            
            synchronizationSvg.append("path")
                .attr("class", "synch-line")
                .attr("fill", "none")
                .attr("stroke", "#ff5722")
                .attr("stroke-width", 2);
        }}
        
        function updateVisualization() {{
            const activationData = animationData.activations[currentIteration][0]; // First batch
            const synchData = animationData.synchronization[currentIteration][0]; // First batch
            
            // Update activations
            const width = 800;
            const height = 300;
            const margin = {{ top: 20, right: 20, bottom: 30, left: 40 }};
            
            const yScale = d3.scaleLinear()
                .domain([-1, 1])
                .range([height - margin.bottom, margin.top]);
            
            activationsSvg.selectAll(".neuron-bar")
                .data(activationData)
                .transition()
                .duration(100)
                .attr("y", d => d >= 0 ? yScale(d) : yScale(0))
                .attr("height", d => Math.abs(yScale(d) - yScale(0)))
                .attr("fill", d => d >= 0 ? "#4caf50" : "#f44336");
            
            // Update synchronization
            const synchYScale = d3.scaleLinear()
                .domain([0, 1])
                .range([height - margin.bottom, margin.top]);
            
            const synchXScale = d3.scaleLinear()
                .domain([0, synchData.length - 1])
                .range([margin.left, width - margin.right]);
            
            const line = d3.line()
                .x((d, i) => synchXScale(i))
                .y(d => synchYScale(d));
            
            synchronizationSvg.select(".synch-line")
                .datum(synchData)
                .transition()
                .duration(100)
                .attr("d", line);
            
            // Update progress and iteration display
            const progress = (currentIteration / (animationData.iterations - 1)) * 100;
            document.getElementById("progress").style.width = progress + "%";
            document.getElementById("current-iteration").textContent = currentIteration;
        }}
        
        function playAnimation() {{
            if (!isPlaying) {{
                isPlaying = true;
                animationInterval = setInterval(() => {{
                    currentIteration = (currentIteration + 1) % animationData.iterations;
                    updateVisualization();
                }}, animationSpeed);
            }}
        }}
        
        function pauseAnimation() {{
            isPlaying = false;
            clearInterval(animationInterval);
        }}
        
        function resetAnimation() {{
            pauseAnimation();
            currentIteration = 0;
            updateVisualization();
        }}
        
        function updateSpeed() {{
            const speed = document.getElementById("speed").value;
            animationSpeed = 1100 - (speed * 100); // Invert so higher value = faster
            if (isPlaying) {{
                pauseAnimation();
                playAnimation();
            }}
        }}
        
        // Initialize on load
        initializeAnimation();
    </script>
</body>
</html>
        """
    
    def _generate_metrics_dashboard_html(self, metric_history: Dict[str, List[float]]) -> str:
        """Generate HTML for real-time metrics dashboard."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Metrics Dashboard</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div class="container">
        <h1>Real-time Metrics Dashboard</h1>
        
        <div class="controls">
            <button onclick="startRealTimeUpdates()">Start Real-time</button>
            <button onclick="stopRealTimeUpdates()">Stop</button>
            <label>
                Update Interval: <input type="range" id="updateInterval" min="100" max="5000" value="1000" onchange="updateInterval()"> ms
            </label>
        </div>
        
        <div class="metric-displays">
            {self._generate_metric_displays(metric_history)}
        </div>
        
        <div class="visualization-panel">
            <h3>Metric Evolution</h3>
            <svg id="metrics-chart" width="800" height="400"></svg>
        </div>
        
        <div class="visualization-panel">
            <h3>Metric Correlations</h3>
            <svg id="correlation-matrix" width="400" height="400"></svg>
        </div>
    </div>

    <script>
        const metricHistory = {json.dumps(metric_history)};
        let realTimeInterval;
        let isRealTime = false;
        
        function initializeDashboard() {{
            setupMetricsChart();
            setupCorrelationMatrix();
            updateVisualizations();
        }}
        
        function setupMetricsChart() {{
            const svg = d3.select("#metrics-chart");
            const width = 800;
            const height = 400;
            const margin = {{ top: 20, right: 80, bottom: 30, left: 50 }};
            
            svg.selectAll("*").remove();
            
            const xScale = d3.scaleLinear()
                .domain([0, Math.max(...Object.values(metricHistory).map(arr => arr.length)) - 1])
                .range([margin.left, width - margin.right]);
            
            const yScale = d3.scaleLinear()
                .domain([0, 1])
                .range([height - margin.bottom, margin.top]);
            
            // Add axes
            svg.append("g")
                .attr("transform", `translate(0,${{height - margin.bottom}})`)
                .call(d3.axisBottom(xScale));
            
            svg.append("g")
                .attr("transform", `translate(${{margin.left}},0)`)
                .call(d3.axisLeft(yScale));
            
            // Add lines for each metric
            const colors = d3.schemeCategory10;
            const line = d3.line()
                .x((d, i) => xScale(i))
                .y(d => yScale(d));
            
            Object.keys(metricHistory).forEach((metric, i) => {{
                svg.append("path")
                    .datum(metricHistory[metric])
                    .attr("class", `metric-line metric-${{metric}}`)
                    .attr("fill", "none")
                    .attr("stroke", colors[i % colors.length])
                    .attr("stroke-width", 2)
                    .attr("d", line);
                
                // Add legend
                svg.append("text")
                    .attr("x", width - margin.right + 10)
                    .attr("y", margin.top + i * 20)
                    .attr("fill", colors[i % colors.length])
                    .text(metric);
            }});
        }}
        
        function setupCorrelationMatrix() {{
            const svg = d3.select("#correlation-matrix");
            const size = 400;
            const margin = 40;
            
            svg.selectAll("*").remove();
            
            const metrics = Object.keys(metricHistory);
            const n = metrics.length;
            const cellSize = (size - 2 * margin) / n;
            
            // Compute correlation matrix
            const correlations = [];
            for (let i = 0; i < n; i++) {{
                correlations[i] = [];
                for (let j = 0; j < n; j++) {{
                    correlations[i][j] = computeCorrelation(
                        metricHistory[metrics[i]], 
                        metricHistory[metrics[j]]
                    );
                }}
            }}
            
            // Create color scale
            const colorScale = d3.scaleSequential(d3.interpolateRdBu)
                .domain([-1, 1]);
            
            // Add cells
            for (let i = 0; i < n; i++) {{
                for (let j = 0; j < n; j++) {{
                    svg.append("rect")
                        .attr("x", margin + j * cellSize)
                        .attr("y", margin + i * cellSize)
                        .attr("width", cellSize)
                        .attr("height", cellSize)
                        .attr("fill", colorScale(correlations[i][j]))
                        .attr("stroke", "white");
                    
                    svg.append("text")
                        .attr("x", margin + j * cellSize + cellSize/2)
                        .attr("y", margin + i * cellSize + cellSize/2)
                        .attr("text-anchor", "middle")
                        .attr("dominant-baseline", "middle")
                        .attr("fill", Math.abs(correlations[i][j]) > 0.5 ? "white" : "black")
                        .text(correlations[i][j].toFixed(2));
                }}
            }}
            
            // Add labels
            metrics.forEach((metric, i) => {{
                svg.append("text")
                    .attr("x", margin + i * cellSize + cellSize/2)
                    .attr("y", margin - 10)
                    .attr("text-anchor", "middle")
                    .text(metric);
                
                svg.append("text")
                    .attr("x", margin - 10)
                    .attr("y", margin + i * cellSize + cellSize/2)
                    .attr("text-anchor", "end")
                    .attr("dominant-baseline", "middle")
                    .text(metric);
            }});
        }}
        
        function computeCorrelation(x, y) {{
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
        
        function updateVisualizations() {{
            // Update metric displays
            Object.keys(metricHistory).forEach(metric => {{
                const latest = metricHistory[metric][metricHistory[metric].length - 1];
                const display = document.getElementById(`metric-${{metric}}`);
                if (display) {{
                    display.textContent = latest.toFixed(3);
                }}
            }});
            
            // Redraw charts
            setupMetricsChart();
            setupCorrelationMatrix();
        }}
        
        function startRealTimeUpdates() {{
            if (!isRealTime) {{
                isRealTime = true;
                const interval = document.getElementById("updateInterval").value;
                realTimeInterval = setInterval(() => {{
                    // Simulate new data
                    Object.keys(metricHistory).forEach(metric => {{
                        const lastValue = metricHistory[metric][metricHistory[metric].length - 1];
                        const newValue = Math.max(0, Math.min(1, lastValue + (Math.random() - 0.5) * 0.1));
                        metricHistory[metric].push(newValue);
                        
                        // Keep only last 100 points
                        if (metricHistory[metric].length > 100) {{
                            metricHistory[metric].shift();
                        }}
                    }});
                    
                    updateVisualizations();
                }}, parseInt(interval));
            }}
        }}
        
        function stopRealTimeUpdates() {{
            isRealTime = false;
            clearInterval(realTimeInterval);
        }}
        
        function updateInterval() {{
            if (isRealTime) {{
                stopRealTimeUpdates();
                startRealTimeUpdates();
            }}
        }}
        
        // Initialize on load
        initializeDashboard();
    </script>
</body>
</html>
        """
    
    def _generate_metric_displays(self, metric_history: Dict[str, List[float]]) -> str:
        """Generate HTML for metric display widgets."""
        displays = []
        for metric_name, values in metric_history.items():
            latest_value = values[-1] if values else 0.0
            displays.append(f"""
            <div class="metric-display">
                <div class="metric-name">{metric_name}</div>
                <div class="metric-value" id="metric-{metric_name}">{latest_value:.3f}</div>
            </div>
            """)
        return "".join(displays)
    
    def _generate_dependency_flow_html(self, dependencies: Dict[str, Any]) -> str:
        """Generate HTML for dependency flow diagram."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Grammar Dependency Flow</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/d3-sankey@0.12.3/dist/d3-sankey.min.js"></script>
    <link rel="stylesheet" href="static/style.css">
</head>
<body>
    <div class="container">
        <h1>Grammar Dependency Flow Diagram</h1>
        
        <div class="controls">
            <label>
                <input type="checkbox" id="showSCC" onchange="toggleSCC()" checked> Show Strongly Connected Components
            </label>
            <label>
                Layout: 
                <select id="layout" onchange="changeLayout()">
                    <option value="sankey">Sankey</option>
                    <option value="hierarchical">Hierarchical</option>
                    <option value="circular">Circular</option>
                </select>
            </label>
        </div>
        
        <div class="visualization-panel">
            <svg id="dependency-svg" width="900" height="600"></svg>
        </div>
        
        <div class="visualization-panel">
            <h3>Strongly Connected Components</h3>
            <div id="scc-info"></div>
        </div>
        
        <div class="tooltip"></div>
    </div>

    <script>
        const dependencies = {json.dumps(dependencies)};
        
        let currentLayout = 'sankey';
        let showSCCs = true;
        
        function initializeDependencyFlow() {{
            updateVisualization();
            displaySCCInfo();
        }}
        
        function updateVisualization() {{
            const svg = d3.select("#dependency-svg");
            svg.selectAll("*").remove();
            
            switch(currentLayout) {{
                case 'sankey':
                    createSankeyDiagram(svg);
                    break;
                case 'hierarchical':
                    createHierarchicalLayout(svg);
                    break;
                case 'circular':
                    createCircularLayout(svg);
                    break;
            }}
        }}
        
        function createSankeyDiagram(svg) {{
            const width = 900;
            const height = 600;
            const margin = {{ top: 20, right: 20, bottom: 20, left: 20 }};
            
            // Prepare data for Sankey
            const sankeyData = {{
                nodes: dependencies.nodes.map(d => ({{ ...d }})),
                links: dependencies.links.map(d => ({{ ...d }}))
            }};
            
            const sankey = d3.sankey()
                .nodeWidth(15)
                .nodePadding(10)
                .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]]);
            
            const {{ nodes, links }} = sankey(sankeyData);
            
            // Add links
            svg.append("g")
                .selectAll("path")
                .data(links)
                .enter().append("path")
                .attr("d", d3.sankeyLinkHorizontal())
                .attr("stroke", d => showSCCs ? d3.schemeCategory10[d.source.group % 10] : "#999")
                .attr("stroke-width", d => Math.max(1, d.width))
                .attr("fill", "none")
                .attr("opacity", 0.5);
            
            // Add nodes
            svg.append("g")
                .selectAll("rect")
                .data(nodes)
                .enter().append("rect")
                .attr("x", d => d.x0)
                .attr("y", d => d.y0)
                .attr("height", d => d.y1 - d.y0)
                .attr("width", d => d.x1 - d.x0)
                .attr("fill", d => showSCCs ? d3.schemeCategory10[d.group % 10] : "#2196f3")
                .on("mouseover", showNodeTooltip)
                .on("mouseout", hideTooltip);
            
            // Add labels
            svg.append("g")
                .selectAll("text")
                .data(nodes)
                .enter().append("text")
                .attr("x", d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
                .attr("y", d => (d.y1 + d.y0) / 2)
                .attr("dy", "0.35em")
                .attr("text-anchor", d => d.x0 < width / 2 ? "start" : "end")
                .text(d => d.id);
        }}
        
        function createHierarchicalLayout(svg) {{
            const width = 900;
            const height = 600;
            
            // Create hierarchy based on topological order
            const levels = {{}};
            dependencies.topo_order.forEach((node, i) => {{
                const level = Math.floor(i / Math.ceil(dependencies.topo_order.length / 5));
                if (!levels[level]) levels[level] = [];
                levels[level].push(node);
            }});
            
            const levelHeight = height / Object.keys(levels).length;
            
            // Position nodes
            const nodePositions = {{}};
            Object.keys(levels).forEach(level => {{
                const levelNodes = levels[level];
                const levelWidth = width / (levelNodes.length + 1);
                levelNodes.forEach((node, i) => {{
                    nodePositions[node] = {{
                        x: (i + 1) * levelWidth,
                        y: (parseInt(level) + 0.5) * levelHeight
                    }};
                }});
            }});
            
            // Draw links
            svg.selectAll(".dep-link")
                .data(dependencies.links)
                .enter().append("line")
                .attr("class", "dep-link")
                .attr("x1", d => nodePositions[d.source].x)
                .attr("y1", d => nodePositions[d.source].y)
                .attr("x2", d => nodePositions[d.target].x)
                .attr("y2", d => nodePositions[d.target].y)
                .attr("stroke", "#999")
                .attr("stroke-width", 2)
                .attr("marker-end", "url(#arrowhead)");
            
            // Add arrowhead marker
            svg.append("defs").append("marker")
                .attr("id", "arrowhead")
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 8)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", "#999");
            
            // Draw nodes
            svg.selectAll(".dep-node")
                .data(dependencies.nodes)
                .enter().append("circle")
                .attr("class", "dep-node")
                .attr("cx", d => nodePositions[d.id].x)
                .attr("cy", d => nodePositions[d.id].y)
                .attr("r", 20)
                .attr("fill", d => showSCCs ? d3.schemeCategory10[d.group % 10] : "#2196f3")
                .on("mouseover", showNodeTooltip)
                .on("mouseout", hideTooltip);
            
            // Add labels
            svg.selectAll(".dep-label")
                .data(dependencies.nodes)
                .enter().append("text")
                .attr("class", "dep-label")
                .attr("x", d => nodePositions[d.id].x)
                .attr("y", d => nodePositions[d.id].y + 35)
                .attr("text-anchor", "middle")
                .text(d => d.id);
        }}
        
        function createCircularLayout(svg) {{
            const width = 900;
            const height = 600;
            const centerX = width / 2;
            const centerY = height / 2;
            const radius = Math.min(width, height) / 2 - 50;
            
            // Position nodes in circle
            const nodePositions = {{}};
            dependencies.nodes.forEach((node, i) => {{
                const angle = (i / dependencies.nodes.length) * 2 * Math.PI;
                nodePositions[node.id] = {{
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle)
                }};
            }});
            
            // Draw links as curves
            svg.selectAll(".dep-link")
                .data(dependencies.links)
                .enter().append("path")
                .attr("class", "dep-link")
                .attr("d", d => {{
                    const source = nodePositions[d.source];
                    const target = nodePositions[d.target];
                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const dr = Math.sqrt(dx * dx + dy * dy) * 0.3;
                    return `M${{source.x}},${{source.y}}A${{dr}},${{dr}} 0 0,1 ${{target.x}},${{target.y}}`;
                }})
                .attr("stroke", "#999")
                .attr("stroke-width", 2)
                .attr("fill", "none")
                .attr("marker-end", "url(#arrowhead)");
            
            // Add arrowhead marker
            svg.append("defs").append("marker")
                .attr("id", "arrowhead")
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 8)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", "#999");
            
            // Draw nodes
            svg.selectAll(".dep-node")
                .data(dependencies.nodes)
                .enter().append("circle")
                .attr("class", "dep-node")
                .attr("cx", d => nodePositions[d.id].x)
                .attr("cy", d => nodePositions[d.id].y)
                .attr("r", 15)
                .attr("fill", d => showSCCs ? d3.schemeCategory10[d.group % 10] : "#2196f3")
                .on("mouseover", showNodeTooltip)
                .on("mouseout", hideTooltip);
            
            // Add labels
            svg.selectAll(".dep-label")
                .data(dependencies.nodes)
                .enter().append("text")
                .attr("class", "dep-label")
                .attr("x", d => nodePositions[d.id].x)
                .attr("y", d => nodePositions[d.id].y + 25)
                .attr("text-anchor", "middle")
                .text(d => d.id);
        }}
        
        function displaySCCInfo() {{
            const sccInfo = document.getElementById("scc-info");
            sccInfo.innerHTML = "";
            
            dependencies.sccs.forEach((scc, i) => {{
                const div = document.createElement("div");
                div.innerHTML = `<strong>Component ${{i + 1}}:</strong> ${{scc.join(", ")}}`;
                div.style.color = d3.schemeCategory10[i % 10];
                div.style.margin = "5px 0";
                sccInfo.appendChild(div);
            }});
        }}
        
        function toggleSCC() {{
            showSCCs = document.getElementById("showSCC").checked;
            updateVisualization();
        }}
        
        function changeLayout() {{
            currentLayout = document.getElementById("layout").value;
            updateVisualization();
        }}
        
        function showNodeTooltip(event, d) {{
            const tooltip = d3.select(".tooltip");
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`Node: ${{d.id}}<br/>Group: ${{d.group}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }}
        
        function hideTooltip() {{
            d3.select(".tooltip").transition().duration(500).style("opacity", 0);
        }}
        
        // Initialize on load
        initializeDependencyFlow();
    </script>
</body>
</html>
        """
    
    def create_comprehensive_dashboard(self, 
                                     udl: UDLRepresentation,
                                     tracking_data: Optional[TrackingData] = None,
                                     metric_history: Optional[Dict[str, List[float]]] = None,
                                     save_path: Optional[str] = None) -> str:
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            udl: UDL representation
            tracking_data: Optional CTM tracking data
            metric_history: Optional metric computation history
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "comprehensive_dashboard.html"
        else:
            save_path = Path(save_path)
        
        # Create individual visualizations
        grammar_graph_path = self.create_interactive_grammar_graph(udl)
        dependency_flow_path = self.create_dependency_flow_diagram(udl)
        
        paths = {
            "grammar_graph": grammar_graph_path,
            "dependency_flow": dependency_flow_path
        }
        
        if tracking_data:
            ctm_animation_path = self.create_ctm_animation(tracking_data)
            paths["ctm_animation"] = ctm_animation_path
        
        if metric_history:
            metrics_dashboard_path = self.create_real_time_metrics_dashboard(metric_history)
            paths["metrics_dashboard"] = metrics_dashboard_path
        
        # Generate comprehensive dashboard HTML
        html_content = self._generate_comprehensive_dashboard_html(paths)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def _generate_comprehensive_dashboard_html(self, visualization_paths: Dict[str, str]) -> str:
        """Generate HTML for comprehensive dashboard."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>UDL Rating Framework - Comprehensive Dashboard</title>
    <link rel="stylesheet" href="static/style.css">
    <style>
        .dashboard-nav {{
            background: #2196f3;
            padding: 10px 0;
            margin-bottom: 20px;
        }}
        
        .dashboard-nav ul {{
            list-style: none;
            margin: 0;
            padding: 0;
            text-align: center;
        }}
        
        .dashboard-nav li {{
            display: inline-block;
            margin: 0 10px;
        }}
        
        .dashboard-nav a {{
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
            transition: background 0.3s;
        }}
        
        .dashboard-nav a:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
        
        .visualization-frame {{
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        
        .dashboard-section {{
            margin: 20px 0;
            display: none;
        }}
        
        .dashboard-section.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UDL Rating Framework - Comprehensive Dashboard</h1>
        
        <nav class="dashboard-nav">
            <ul>
                <li><a href="#" onclick="showSection('grammar-graph')">Grammar Graph</a></li>
                <li><a href="#" onclick="showSection('dependency-flow')">Dependency Flow</a></li>
                {"<li><a href='#' onclick=\"showSection('ctm-animation')\">CTM Animation</a></li>" if 'ctm_animation' in visualization_paths else ""}
                {"<li><a href='#' onclick=\"showSection('metrics-dashboard')\">Metrics Dashboard</a></li>" if 'metrics_dashboard' in visualization_paths else ""}
            </ul>
        </nav>
        
        <div id="grammar-graph" class="dashboard-section active">
            <h2>Interactive Grammar Graph</h2>
            <iframe src="{os.path.basename(visualization_paths['grammar_graph'])}" class="visualization-frame"></iframe>
        </div>
        
        <div id="dependency-flow" class="dashboard-section">
            <h2>Grammar Dependency Flow</h2>
            <iframe src="{os.path.basename(visualization_paths['dependency_flow'])}" class="visualization-frame"></iframe>
        </div>
        
        {"<div id='ctm-animation' class='dashboard-section'><h2>CTM Processing Animation</h2><iframe src='" + os.path.basename(visualization_paths['ctm_animation']) + "' class='visualization-frame'></iframe></div>" if 'ctm_animation' in visualization_paths else ""}
        
        {"<div id='metrics-dashboard' class='dashboard-section'><h2>Real-time Metrics Dashboard</h2><iframe src='" + os.path.basename(visualization_paths['metrics_dashboard']) + "' class='visualization-frame'></iframe></div>" if 'metrics_dashboard' in visualization_paths else ""}
    </div>

    <script>
        function showSection(sectionId) {{
            // Hide all sections
            const sections = document.querySelectorAll('.dashboard-section');
            sections.forEach(section => section.classList.remove('active'));
            
            // Show selected section
            document.getElementById(sectionId).classList.add('active');
        }}
    </script>
</body>
</html>
        """