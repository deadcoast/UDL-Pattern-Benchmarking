"""
WebGL-based 3D Visualization Utilities.

Provides advanced 3D visualizations using WebGL for grammar graphs
and CTM processing with enhanced interactivity.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from ..core.representation import UDLRepresentation
from ..models.ctm_adapter import TrackingData


class WebGLVisualizer:
    """
    Creates advanced 3D visualizations using WebGL.
    
    Provides high-performance 3D rendering for:
    - Complex grammar graphs with thousands of nodes
    - Real-time CTM processing visualization
    - Interactive exploration of language structures
    """
    
    def __init__(self, output_dir: str = "webgl_visualizations"):
        """Initialize WebGL visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._setup_webgl_assets()
    
    def _setup_webgl_assets(self):
        """Set up WebGL shaders and assets."""
        shaders_dir = self.output_dir / "shaders"
        shaders_dir.mkdir(exist_ok=True)
        
        # Vertex shader for 3D nodes
        vertex_shader = """
        attribute vec3 position;
        attribute vec3 color;
        attribute float size;
        
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform float time;
        
        varying vec3 vColor;
        varying float vSize;
        
        void main() {
            vColor = color;
            vSize = size;
            
            vec3 pos = position;
            // Add subtle animation
            pos.y += sin(time + position.x) * 0.1;
            
            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
            gl_PointSize = size;
        }
        """
        
        # Fragment shader for 3D nodes
        fragment_shader = """
        precision mediump float;
        
        varying vec3 vColor;
        varying float vSize;
        
        void main() {
            vec2 center = gl_PointCoord - vec2(0.5);
            float dist = length(center);
            
            if (dist > 0.5) {
                discard;
            }
            
            float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
            gl_FragColor = vec4(vColor, alpha);
        }
        """
        
        with open(shaders_dir / "vertex.glsl", "w") as f:
            f.write(vertex_shader)
        
        with open(shaders_dir / "fragment.glsl", "w") as f:
            f.write(fragment_shader)
    
    def create_3d_grammar_graph(self, 
                               udl: UDLRepresentation,
                               save_path: Optional[str] = None) -> str:
        """
        Create advanced 3D grammar graph with WebGL.
        
        Args:
            udl: UDL representation with grammar graph
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "3d_grammar_graph.html"
        else:
            save_path = Path(save_path)
        
        # Extract and process graph data
        graph = udl.get_grammar_graph()
        graph_data = self._process_graph_for_3d(graph)
        
        # Generate WebGL HTML
        html_content = self._generate_3d_graph_html(graph_data)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def create_ctm_3d_animation(self, 
                               tracking_data: TrackingData,
                               save_path: Optional[str] = None) -> str:
        """
        Create 3D animation of CTM processing.
        
        Args:
            tracking_data: CTM tracking data
            save_path: Optional path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        if save_path is None:
            save_path = self.output_dir / "ctm_3d_animation.html"
        else:
            save_path = Path(save_path)
        
        # Process tracking data for 3D visualization
        animation_data = self._process_tracking_data_for_3d(tracking_data)
        
        # Generate WebGL HTML
        html_content = self._generate_ctm_3d_html(animation_data)
        
        with open(save_path, "w") as f:
            f.write(html_content)
        
        return str(save_path)
    
    def _process_graph_for_3d(self, graph) -> Dict[str, Any]:
        """Process NetworkX graph for 3D visualization."""
        import networkx as nx
        
        # Use spring layout in 3D
        pos_2d = nx.spring_layout(graph, k=3, iterations=50)
        
        # Extend to 3D by adding z-coordinate based on node properties
        nodes_3d = []
        for node_id, node_data in graph.nodes(data=True):
            x, y = pos_2d[node_id]
            # Z-coordinate based on node degree or type
            z = len(list(graph.neighbors(node_id))) * 0.5
            
            nodes_3d.append({
                "id": str(node_id),
                "position": [x * 10, y * 10, z],
                "color": self._get_node_color_rgb(node_data.get("type", "unknown")),
                "size": max(5, len(str(node_id)) * 2),
                "type": node_data.get("type", "unknown")
            })
        
        # Process edges
        edges_3d = []
        for source, target, edge_data in graph.edges(data=True):
            source_pos = next(n["position"] for n in nodes_3d if n["id"] == str(source))
            target_pos = next(n["position"] for n in nodes_3d if n["id"] == str(target))
            
            edges_3d.append({
                "source": str(source),
                "target": str(target),
                "source_pos": source_pos,
                "target_pos": target_pos,
                "weight": edge_data.get("weight", 1)
            })
        
        return {
            "nodes": nodes_3d,
            "edges": edges_3d,
            "bounds": self._calculate_bounds(nodes_3d)
        }
    
    def _process_tracking_data_for_3d(self, tracking_data: TrackingData) -> Dict[str, Any]:
        """Process tracking data for 3D CTM visualization."""
        # Create 3D neuron positions
        n_neurons = tracking_data.n_neurons
        neurons_3d = []
        
        # Arrange neurons in a 3D grid
        grid_size = int(np.ceil(n_neurons ** (1/3)))
        for i in range(n_neurons):
            x = (i % grid_size) - grid_size // 2
            y = ((i // grid_size) % grid_size) - grid_size // 2
            z = (i // (grid_size * grid_size)) - grid_size // 2
            
            neurons_3d.append({
                "id": i,
                "position": [x * 2, y * 2, z * 2],
                "activations": tracking_data.post_activations[:, 0, i].tolist()  # First batch
            })
        
        return {
            "neurons": neurons_3d,
            "iterations": tracking_data.iterations,
            "synchronization": tracking_data.synch_out[:, 0, :].tolist()  # First batch
        }
    
    def _get_node_color_rgb(self, node_type: str) -> List[float]:
        """Get RGB color for node type."""
        color_map = {
            "terminal": [0.3, 0.7, 0.3],      # Green
            "non_terminal": [0.1, 0.6, 0.9],  # Blue
            "rule": [1.0, 0.6, 0.0],          # Orange
            "constraint": [0.9, 0.3, 0.3],    # Red
            "unknown": [0.6, 0.6, 0.6]        # Gray
        }
        return color_map.get(node_type, [0.6, 0.6, 0.6])
    
    def _calculate_bounds(self, nodes_3d: List[Dict]) -> Dict[str, List[float]]:
        """Calculate 3D bounds for camera positioning."""
        positions = [node["position"] for node in nodes_3d]
        if not positions:
            return {"min": [0, 0, 0], "max": [1, 1, 1]}
        
        positions_array = np.array(positions)
        return {
            "min": positions_array.min(axis=0).tolist(),
            "max": positions_array.max(axis=0).tolist()
        }
    
    def _generate_3d_graph_html(self, graph_data: Dict[str, Any]) -> str:
        """Generate HTML for 3D grammar graph visualization."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Grammar Graph Visualization</title>
    <script src="https://unpkg.com/three@0.147.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.147.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #000; font-family: Arial, sans-serif; }}
        #container {{ position: relative; width: 100vw; height: 100vh; }}
        #controls {{ 
            position: absolute; 
            top: 10px; 
            left: 10px; 
            background: rgba(0,0,0,0.7); 
            color: white; 
            padding: 15px; 
            border-radius: 5px;
            z-index: 100;
        }}
        #info {{ 
            position: absolute; 
            bottom: 10px; 
            left: 10px; 
            background: rgba(0,0,0,0.7); 
            color: white; 
            padding: 10px; 
            border-radius: 5px;
            z-index: 100;
        }}
        button {{ 
            margin: 5px; 
            padding: 8px 12px; 
            background: #2196f3; 
            color: white; 
            border: none; 
            border-radius: 3px; 
            cursor: pointer; 
        }}
        button:hover {{ background: #1976d2; }}
        input[type="range"] {{ width: 150px; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h3>3D Grammar Graph Controls</h3>
            <button onclick="resetCamera()">Reset Camera</button>
            <button onclick="toggleAnimation()">Toggle Animation</button>
            <br>
            <label>Node Size: <input type="range" id="nodeSize" min="0.5" max="3" step="0.1" value="1" onchange="updateNodeSize()"></label>
            <br>
            <label>Animation Speed: <input type="range" id="animSpeed" min="0.1" max="2" step="0.1" value="1" onchange="updateAnimSpeed()"></label>
            <br>
            <label>
                <input type="checkbox" id="showEdges" checked onchange="toggleEdges()"> Show Edges
            </label>
            <br>
            <label>
                <input type="checkbox" id="showLabels" checked onchange="toggleLabels()"> Show Labels
            </label>
        </div>
        
        <div id="info">
            <div>Nodes: {len(graph_data['nodes'])}</div>
            <div>Edges: {len(graph_data['edges'])}</div>
            <div>FPS: <span id="fps">0</span></div>
        </div>
    </div>

    <script>
        const graphData = {json.dumps(graph_data)};
        
        let scene, camera, renderer, controls;
        let nodeGroup, edgeGroup, labelGroup;
        let animationEnabled = true;
        let animationSpeed = 1;
        let nodeScale = 1;
        let showEdges = true;
        let showLabels = true;
        
        // Performance monitoring
        let frameCount = 0;
        let lastTime = performance.now();
        
        function init() {{
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            
            // Create camera
            const bounds = graphData.bounds;
            const center = [
                (bounds.min[0] + bounds.max[0]) / 2,
                (bounds.min[1] + bounds.max[1]) / 2,
                (bounds.min[2] + bounds.max[2]) / 2
            ];
            const size = Math.max(
                bounds.max[0] - bounds.min[0],
                bounds.max[1] - bounds.min[1],
                bounds.max[2] - bounds.min[2]
            );
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(center[0], center[1], center[2] + size * 2);
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Create controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(center[0], center[1], center[2]);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(50, 50, 50);
            directionalLight.castShadow = true;
            directionalLight.shadow.mapSize.width = 2048;
            directionalLight.shadow.mapSize.height = 2048;
            scene.add(directionalLight);
            
            // Create node group
            nodeGroup = new THREE.Group();
            scene.add(nodeGroup);
            
            // Create edge group
            edgeGroup = new THREE.Group();
            scene.add(edgeGroup);
            
            // Create label group
            labelGroup = new THREE.Group();
            scene.add(labelGroup);
            
            // Create nodes
            createNodes();
            
            // Create edges
            createEdges();
            
            // Create labels
            createLabels();
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize, false);
            
            // Start animation loop
            animate();
        }}
        
        function createNodes() {{
            graphData.nodes.forEach(node => {{
                const geometry = new THREE.SphereGeometry(node.size * 0.1, 16, 16);
                const material = new THREE.MeshPhongMaterial({{
                    color: new THREE.Color(node.color[0], node.color[1], node.color[2]),
                    shininess: 100
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(node.position[0], node.position[1], node.position[2]);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                mesh.userData = {{ nodeId: node.id, nodeType: node.type }};
                
                nodeGroup.add(mesh);
            }});
        }}
        
        function createEdges() {{
            graphData.edges.forEach(edge => {{
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array([
                    edge.source_pos[0], edge.source_pos[1], edge.source_pos[2],
                    edge.target_pos[0], edge.target_pos[1], edge.target_pos[2]
                ]);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                const material = new THREE.LineBasicMaterial({{
                    color: 0x666666,
                    opacity: 0.6,
                    transparent: true
                }});
                
                const line = new THREE.Line(geometry, material);
                edgeGroup.add(line);
            }});
        }}
        
        function createLabels() {{
            const loader = new THREE.FontLoader();
            // For simplicity, we'll create simple text sprites instead of 3D text
            graphData.nodes.forEach(node => {{
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 128;
                canvas.height = 64;
                
                context.fillStyle = 'rgba(0, 0, 0, 0.8)';
                context.fillRect(0, 0, canvas.width, canvas.height);
                
                context.fillStyle = 'white';
                context.font = '16px Arial';
                context.textAlign = 'center';
                context.fillText(node.id, canvas.width / 2, canvas.height / 2 + 6);
                
                const texture = new THREE.CanvasTexture(canvas);
                const material = new THREE.SpriteMaterial({{ map: texture }});
                const sprite = new THREE.Sprite(material);
                
                sprite.position.set(
                    node.position[0], 
                    node.position[1] + node.size * 0.2, 
                    node.position[2]
                );
                sprite.scale.set(2, 1, 1);
                
                labelGroup.add(sprite);
            }});
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            
            // Update controls
            controls.update();
            
            // Animation effects
            if (animationEnabled) {{
                const time = Date.now() * 0.001 * animationSpeed;
                
                // Animate nodes with subtle floating motion
                nodeGroup.children.forEach((node, index) => {{
                    const originalY = graphData.nodes[index].position[1];
                    node.position.y = originalY + Math.sin(time + index * 0.5) * 0.2;
                    
                    // Pulse effect based on node type
                    const scale = 1 + Math.sin(time * 2 + index) * 0.1;
                    node.scale.setScalar(scale * nodeScale);
                }});
                
                // Animate edges with flowing effect
                edgeGroup.children.forEach((edge, index) => {{
                    const material = edge.material;
                    material.opacity = 0.3 + Math.sin(time * 3 + index) * 0.3;
                }});
            }}
            
            // Update FPS counter
            frameCount++;
            const currentTime = performance.now();
            if (currentTime - lastTime >= 1000) {{
                document.getElementById('fps').textContent = Math.round(frameCount * 1000 / (currentTime - lastTime));
                frameCount = 0;
                lastTime = currentTime;
            }}
            
            // Render
            renderer.render(scene, camera);
        }}
        
        function resetCamera() {{
            const bounds = graphData.bounds;
            const center = [
                (bounds.min[0] + bounds.max[0]) / 2,
                (bounds.min[1] + bounds.max[1]) / 2,
                (bounds.min[2] + bounds.max[2]) / 2
            ];
            const size = Math.max(
                bounds.max[0] - bounds.min[0],
                bounds.max[1] - bounds.min[1],
                bounds.max[2] - bounds.min[2]
            );
            
            camera.position.set(center[0], center[1], center[2] + size * 2);
            controls.target.set(center[0], center[1], center[2]);
            controls.update();
        }}
        
        function toggleAnimation() {{
            animationEnabled = !animationEnabled;
        }}
        
        function updateNodeSize() {{
            nodeScale = parseFloat(document.getElementById('nodeSize').value);
        }}
        
        function updateAnimSpeed() {{
            animationSpeed = parseFloat(document.getElementById('animSpeed').value);
        }}
        
        function toggleEdges() {{
            showEdges = document.getElementById('showEdges').checked;
            edgeGroup.visible = showEdges;
        }}
        
        function toggleLabels() {{
            showLabels = document.getElementById('showLabels').checked;
            labelGroup.visible = showLabels;
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        // Initialize when page loads
        init();
    </script>
</body>
</html>
        """
    
    def _generate_ctm_3d_html(self, animation_data: Dict[str, Any]) -> str:
        """Generate HTML for 3D CTM animation."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D CTM Processing Animation</title>
    <script src="https://unpkg.com/three@0.147.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.147.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #000; font-family: Arial, sans-serif; }}
        #container {{ position: relative; width: 100vw; height: 100vh; }}
        #controls {{ 
            position: absolute; 
            top: 10px; 
            left: 10px; 
            background: rgba(0,0,0,0.8); 
            color: white; 
            padding: 15px; 
            border-radius: 5px;
            z-index: 100;
            max-width: 300px;
        }}
        #timeline {{ 
            position: absolute; 
            bottom: 10px; 
            left: 10px; 
            right: 10px; 
            background: rgba(0,0,0,0.8); 
            color: white; 
            padding: 15px; 
            border-radius: 5px;
            z-index: 100;
        }}
        button {{ 
            margin: 5px; 
            padding: 8px 12px; 
            background: #2196f3; 
            color: white; 
            border: none; 
            border-radius: 3px; 
            cursor: pointer; 
        }}
        button:hover {{ background: #1976d2; }}
        input[type="range"] {{ width: 200px; }}
        .timeline-bar {{ 
            width: 100%; 
            height: 20px; 
            background: #333; 
            border-radius: 10px; 
            overflow: hidden; 
            margin: 10px 0; 
        }}
        .timeline-progress {{ 
            height: 100%; 
            background: linear-gradient(90deg, #4caf50, #2196f3); 
            transition: width 0.1s ease; 
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h3>3D CTM Animation</h3>
            <button onclick="playPause()">Play/Pause</button>
            <button onclick="reset()">Reset</button>
            <button onclick="stepForward()">Step Forward</button>
            <button onclick="stepBackward()">Step Backward</button>
            <br>
            <label>Speed: <input type="range" id="speed" min="0.1" max="3" step="0.1" value="1" onchange="updateSpeed()"></label>
            <br>
            <label>Neuron Size: <input type="range" id="neuronSize" min="0.5" max="3" step="0.1" value="1" onchange="updateNeuronSize()"></label>
            <br>
            <label>
                <input type="checkbox" id="showConnections" onchange="toggleConnections()"> Show Connections
            </label>
            <br>
            <label>
                <input type="checkbox" id="showSynchronization" checked onchange="toggleSynchronization()"> Show Synchronization
            </label>
            <br>
            <div>Iteration: <span id="currentIteration">0</span> / {animation_data['iterations'] - 1}</div>
        </div>
        
        <div id="timeline">
            <div>Timeline Control</div>
            <div class="timeline-bar">
                <div class="timeline-progress" id="timelineProgress"></div>
            </div>
            <input type="range" id="timelineSlider" min="0" max="{animation_data['iterations'] - 1}" value="0" onchange="seekToIteration()">
        </div>
    </div>

    <script>
        const animationData = {json.dumps(animation_data)};
        
        let scene, camera, renderer, controls;
        let neuronGroup, connectionGroup, synchronizationGroup;
        let currentIteration = 0;
        let isPlaying = false;
        let animationSpeed = 1;
        let neuronScale = 1;
        let showConnections = false;
        let showSynchronization = true;
        let playInterval;
        
        function init() {{
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0a);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 20);
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Create controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(10, 10, 10);
            scene.add(directionalLight);
            
            // Create groups
            neuronGroup = new THREE.Group();
            scene.add(neuronGroup);
            
            connectionGroup = new THREE.Group();
            scene.add(connectionGroup);
            
            synchronizationGroup = new THREE.Group();
            scene.add(synchronizationGroup);
            
            // Create neurons
            createNeurons();
            
            // Create connections
            createConnections();
            
            // Create synchronization visualization
            createSynchronizationVisualization();
            
            // Update initial state
            updateVisualization();
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize, false);
            
            // Start render loop
            animate();
        }}
        
        function createNeurons() {{
            animationData.neurons.forEach((neuron, index) => {{
                const geometry = new THREE.SphereGeometry(0.2, 16, 16);
                const material = new THREE.MeshPhongMaterial({{
                    color: 0x2196f3,
                    transparent: true,
                    opacity: 0.8
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(neuron.position[0], neuron.position[1], neuron.position[2]);
                mesh.userData = {{ neuronId: neuron.id }};
                
                neuronGroup.add(mesh);
            }});
        }}
        
        function createConnections() {{
            // Create connections between nearby neurons
            const neurons = animationData.neurons;
            for (let i = 0; i < neurons.length; i++) {{
                for (let j = i + 1; j < neurons.length; j++) {{
                    const pos1 = neurons[i].position;
                    const pos2 = neurons[j].position;
                    const distance = Math.sqrt(
                        Math.pow(pos1[0] - pos2[0], 2) +
                        Math.pow(pos1[1] - pos2[1], 2) +
                        Math.pow(pos1[2] - pos2[2], 2)
                    );
                    
                    // Only connect nearby neurons
                    if (distance < 4) {{
                        const geometry = new THREE.BufferGeometry();
                        const positions = new Float32Array([
                            pos1[0], pos1[1], pos1[2],
                            pos2[0], pos2[1], pos2[2]
                        ]);
                        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        
                        const material = new THREE.LineBasicMaterial({{
                            color: 0x666666,
                            opacity: 0.3,
                            transparent: true
                        }});
                        
                        const line = new THREE.Line(geometry, material);
                        connectionGroup.add(line);
                    }}
                }}
            }}
            
            connectionGroup.visible = showConnections;
        }}
        
        function createSynchronizationVisualization() {{
            // Create a visual representation of synchronization values
            const synchData = animationData.synchronization[0]; // First iteration
            synchData.forEach((value, index) => {{
                const geometry = new THREE.RingGeometry(2 + index * 0.5, 2.2 + index * 0.5, 32);
                const material = new THREE.MeshBasicMaterial({{
                    color: 0xff5722,
                    transparent: true,
                    opacity: 0.5,
                    side: THREE.DoubleSide
                }});
                
                const ring = new THREE.Mesh(geometry, material);
                ring.position.set(0, 0, -5);
                ring.userData = {{ synchIndex: index }};
                
                synchronizationGroup.add(ring);
            }});
        }}
        
        function updateVisualization() {{
            // Update neuron colors and sizes based on activations
            neuronGroup.children.forEach((neuron, index) => {{
                const activation = animationData.neurons[index].activations[currentIteration];
                
                // Color based on activation (blue to red)
                const intensity = Math.abs(activation);
                const color = activation >= 0 
                    ? new THREE.Color(0.2, 0.6 + intensity * 0.4, 0.2)  // Green for positive
                    : new THREE.Color(0.8, 0.2, 0.2);  // Red for negative
                
                neuron.material.color = color;
                
                // Size based on activation magnitude
                const scale = neuronScale * (0.5 + intensity * 1.5);
                neuron.scale.setScalar(scale);
                
                // Opacity based on activation
                neuron.material.opacity = 0.3 + intensity * 0.7;
            }});
            
            // Update synchronization visualization
            if (showSynchronization && animationData.synchronization[currentIteration]) {{
                const synchData = animationData.synchronization[currentIteration];
                synchronizationGroup.children.forEach((ring, index) => {{
                    if (index < synchData.length) {{
                        const value = synchData[index];
                        ring.material.opacity = value * 0.8;
                        ring.rotation.z += value * 0.1;
                        
                        // Color based on synchronization value
                        const hue = value * 0.7; // 0 to 0.7 (red to blue)
                        ring.material.color.setHSL(hue, 1, 0.5);
                    }}
                }});
            }}
            
            // Update UI
            document.getElementById('currentIteration').textContent = currentIteration;
            const progress = (currentIteration / (animationData.iterations - 1)) * 100;
            document.getElementById('timelineProgress').style.width = progress + '%';
            document.getElementById('timelineSlider').value = currentIteration;
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        function playPause() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    currentIteration = (currentIteration + 1) % animationData.iterations;
                    updateVisualization();
                }}, 200 / animationSpeed);
            }} else {{
                clearInterval(playInterval);
            }}
        }}
        
        function reset() {{
            isPlaying = false;
            clearInterval(playInterval);
            currentIteration = 0;
            updateVisualization();
        }}
        
        function stepForward() {{
            currentIteration = Math.min(currentIteration + 1, animationData.iterations - 1);
            updateVisualization();
        }}
        
        function stepBackward() {{
            currentIteration = Math.max(currentIteration - 1, 0);
            updateVisualization();
        }}
        
        function seekToIteration() {{
            currentIteration = parseInt(document.getElementById('timelineSlider').value);
            updateVisualization();
        }}
        
        function updateSpeed() {{
            animationSpeed = parseFloat(document.getElementById('speed').value);
            if (isPlaying) {{
                clearInterval(playInterval);
                playInterval = setInterval(() => {{
                    currentIteration = (currentIteration + 1) % animationData.iterations;
                    updateVisualization();
                }}, 200 / animationSpeed);
            }}
        }}
        
        function updateNeuronSize() {{
            neuronScale = parseFloat(document.getElementById('neuronSize').value);
            updateVisualization();
        }}
        
        function toggleConnections() {{
            showConnections = document.getElementById('showConnections').checked;
            connectionGroup.visible = showConnections;
        }}
        
        function toggleSynchronization() {{
            showSynchronization = document.getElementById('showSynchronization').checked;
            synchronizationGroup.visible = showSynchronization;
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        // Initialize when page loads
        init();
    </script>
</body>
</html>
        """