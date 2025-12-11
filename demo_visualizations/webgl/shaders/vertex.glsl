
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
        