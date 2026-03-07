/**
 * 3D Hand Tracking - Three.js + MediaPipe Tasks Vision + Vite
 * Modern HandLandmarker API with GPU Acceleration
 * 
 * FEATURES:
 * ✓ Modern @mediapipe/tasks-vision with GPU acceleration
 * ✓ Real-time hand landmark detection (21 points)
 * ✓ Smooth Lerp interpolation for jitter-free tracking
 * ✓ Visible mirrored webcam feed for debugging
 * ✓ 3D sphere following index finger tip
 * ✓ Console logging for validation
 */

import * as THREE from 'three';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ==================== CONFIG ====================
const CONFIG = {
    lerpFactor: 0.1,           // Slightly slower for more dramatic effect
    sphereRadius: 25,
    depthScale: 150,
    // Pinch-to-scale config
    scaleMultiplier: 40,       // Large multiplier for bigger scale differences
    scaleMin: 0.2,             // Tiny point when fully pinched
    scaleMax: 15,              // Massive, screen-filling when fully spread
};

// ==================== STATE ====================
let scene, camera, renderer;
let sphere = null;
let pointLight = null;
let handLandmarker = null;
let isRunning = false;
let animationId = null;
let lastVideoTime = -1;

// Materials for opacity control
let wireframeMaterial = null;
let overlayMaterial = null;

let spherePos = { x: 0, y: 0, z: 0 };
let targetPos = { x: 0, y: 0, z: 0 };

// Scale tracking (for pinch gesture)
let sphereScale = 1.0;
let targetScale = 1.0;

// Stats
let frameCount = 0;
let lastFrameTime = Date.now();
let fps = 0;
let handsDetected = 0;
let frameCounter = 0;
let lastConfidence = 0;

// ==================== UTILITY: LERP ====================
/**
 * Linear interpolation (smooth movement)
 * Formula: pos += (target - pos) * 0.15
 */
function lerp(current, target, factor) {
    return current + (target - current) * factor;
}

/**
 * Map MediaPipe coordinates (0-1) to Three.js world space
 * 
 * CORRECTED FORMULAS:
 * x: (landmark.x - 0.5) * 2 * viewportWidth
 * y: -(landmark.y - 0.5) * 2 * viewportHeight
 */
function mapCoordinates(x, y, z) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    
    return {
        x: (x - 0.5) * 2 * (vw / 2),      // Account for viewport width
        y: -(y - 0.5) * 2 * (vh / 2),     // Invert Y and account for height
        z: z * CONFIG.depthScale - 50      // Depth from hand distance
    };
}

/**
 * Calculate depth from hand landmarks
 * Wrist to finger distance tells us depth (Z-axis)
 */
function calculateDepth(wrist, finger) {
    const dx = wrist.x - finger.x;
    const dy = wrist.y - finger.y;
    const dz = (wrist.z || 0) - (finger.z || 0);
    return Math.sqrt(dx * dx + dy * dy + (dz + 0.5) * (dz + 0.5));
}

// ==================== THREE.JS INITIALIZATION ====================

function initThreeJS() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Create camera with proper aspect ratio
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera = new THREE.PerspectiveCamera(
        75,
        width / height,
        0.1,
        2000
    );
    // Center the camera - sphere starts centered on screen
    camera.position.set(0, 0, 350);
    camera.lookAt(0, 0, 0);

    // Create renderer with full viewport
    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        precision: 'highp'
    });
    
    // Set size to match viewport exactly
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap at 2x for mobile performance
    renderer.shadowMap.enabled = true;
    renderer.domElement.style.position = 'fixed';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.style.width = '100vw';
    renderer.domElement.style.height = '100vh';
    renderer.domElement.style.display = 'block';
    document.getElementById('app').appendChild(renderer.domElement);

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    pointLight = new THREE.PointLight(0x00ffff, 2.5, 600);
    pointLight.position.set(0, 0, 100);
    pointLight.castShadow = true;
    scene.add(pointLight);

    // Create sphere (the glowing cursor)
    createGlowingSphere();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    console.log('✓ Three.js scene initialized');
    console.log('  Viewport:', window.innerWidth, 'x', window.innerHeight);
}

/**
 * Create the glowing wireframe sphere with spider-web effect
 * Mobile optimized: reduced geometry detail for better performance
 */
function createGlowingSphere() {
    // Optimized IcosahedronGeometry for mobile (subdivisions=4)
    // High-detail version was 6, reduced to 4 for better mobile performance
    const geometry = new THREE.IcosahedronGeometry(CONFIG.sphereRadius, 4);

    // Material with wireframe for spider-web effect
    wireframeMaterial = new THREE.MeshStandardMaterial({
        color: 0x00ff00,           // Neon green base
        emissive: 0x00ff00,        // Neon green glow
        emissiveIntensity: 0.9,
        metalness: 0.8,
        roughness: 0.1,
        wireframe: true,           // SPIDER-WEB EFFECT
        wireframeLinewidth: 2
    });

    sphere = new THREE.Mesh(geometry, wireframeMaterial);
    sphere.castShadow = true;
    sphere.receiveShadow = true;
    sphere.position.set(0, 0, 0);
    sphere.scale.set(1, 1, 1);

    // Add solid overlay for more visual pop (semi-transparent)
    const solidGeometry = new THREE.IcosahedronGeometry(CONFIG.sphereRadius - 2, 4);
    overlayMaterial = new THREE.MeshStandardMaterial({
        color: 0x00ffff,           // Cyan overlay
        emissive: 0x00ffff,
        emissiveIntensity: 0.5,
        metalness: 0.4,
        roughness: 0.3,
        transparent: true,
        opacity: 0.2,              // Subtle transparency initially
        wireframe: false
    });

    const solidMesh = new THREE.Mesh(solidGeometry, overlayMaterial);
    solidMesh.castShadow = true;

    const group = new THREE.Group();
    group.add(sphere);
    group.add(solidMesh);
    scene.add(group);

    console.log('✓ Spider-web sphere created (neon green wireframe + cyan overlay)');
}

function onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    // Update camera
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    
    // Update renderer
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    console.log('✓ Window resized:', width, 'x', height, '(DPR:', window.devicePixelRatio.toFixed(1) + ')');
}

// ==================== MEDIAPIPE HANDLANDMARKER INITIALIZATION ====================

async function initMediaPipe() {
    try {
        console.log('Loading MediaPipe HandLandmarker with GPU acceleration...');
        
        // Load WASM files and initialize FilesetResolver
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        // Create HandLandmarker with GPU delegate and VIDEO running mode
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                delegate: 'GPU',
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        console.log('✓ HandLandmarker initialized with GPU acceleration');
        return true;

    } catch (error) {
        console.error('✗ MediaPipe initialization failed:', error);
        throw error;
    }
}

/**
 * Process hand detection results from HandLandmarker
 */
function processHandResults(results) {
    if (!results.landmarks || results.landmarks.length === 0) {
        if (handsDetected > 0) {
            console.log('Hand lost');
        }
        handsDetected = 0;
        lastConfidence = 0;
        return;
    }

    // Track first hand (index 0)
    handsDetected = results.landmarks.length;
    const landmarks = results.landmarks[0]; // Get first hand's landmarks

    if (!landmarks || landmarks.length < 21) {
        console.warn('Incomplete landmarks:', landmarks ? landmarks.length : 0);
        return;
    }

    // Landmark 8 = Index Finger Tip (cursor position)
    const indexTip = landmarks[8];
    const wrist = landmarks[0];
    const middleFinger = landmarks[9];
    
    // Landmark 4 = Thumb Tip (for pinch detection)
    const thumbTip = landmarks[4];

    // Calculate confidence from presence score
    const presence = (indexTip.presence || 0.5);
    lastConfidence = Math.round(presence * 100);

    // ===== PINCH DETECTION (Thumb to Index distance) =====
    // Calculate Euclidean distance between thumb tip and index tip
    const pinchDistX = indexTip.x - thumbTip.x;
    const pinchDistY = indexTip.y - thumbTip.y;
    const pinchDistZ = (indexTip.z || 0) - (thumbTip.z || 0);
    
    const pinchDistance = Math.sqrt(
        pinchDistX * pinchDistX + 
        pinchDistY * pinchDistY + 
        pinchDistZ * pinchDistZ
    );

    // Map pinch distance to scale using multiplier
    // Distance * multiplier gives us the scale
    let calculatedScale = pinchDistance * CONFIG.scaleMultiplier;
    
    // Clamp between min and max
    targetScale = Math.min(Math.max(calculatedScale, CONFIG.scaleMin), CONFIG.scaleMax);

    // ===== DEPTH CALCULATION (Average Z of all landmarks) =====
    // Calculate average Z position of hand landmarks for depth
    let sumZ = 0;
    for (let i = 0; i < landmarks.length; i++) {
        sumZ += (landmarks[i].z || 0);
    }
    const avgZ = sumZ / landmarks.length;

    frameCounter++;

    // DEBUG: Log hand data
    console.log('Hand detected:', handsDetected);
    console.log('  Index Tip (raw):', {
        x: indexTip.x.toFixed(3),
        y: indexTip.y.toFixed(3),
        z: indexTip.z.toFixed(3)
    });

    // Map to 3D coordinates
    targetPos = mapCoordinates(indexTip.x, indexTip.y, avgZ);

    // DEBUG: Log mapped coordinates and scale
    console.log('  Index Tip (mapped):', {
        x: targetPos.x.toFixed(1),
        y: targetPos.y.toFixed(1),
        z: targetPos.z.toFixed(1)
    });
    console.log('  Pinch Distance:', pinchDistance.toFixed(3), '→ Calculated Scale:', calculatedScale.toFixed(2), '→ Clamped:', targetScale.toFixed(2));
}

// ==================== ANIMATION LOOP ====================

/**
 * Update sphere position and scale with smooth LERP interpolation
 */
function updateSphere() {
    if (!sphere) return;

    // ===== POSITION: Apply LERP smoothing =====
    spherePos.x = lerp(spherePos.x, targetPos.x, CONFIG.lerpFactor);
    spherePos.y = lerp(spherePos.y, targetPos.y, CONFIG.lerpFactor);
    spherePos.z = lerp(spherePos.z, targetPos.z, CONFIG.lerpFactor);

    // Update sphere position
    sphere.parent.position.set(spherePos.x, spherePos.y, spherePos.z);

    // ===== SCALE: Apply LERP smoothing to pinch gesture =====
    sphereScale = lerp(sphereScale, targetScale, CONFIG.lerpFactor);
    sphere.parent.scale.set(sphereScale, sphereScale, sphereScale);

    // ===== OPACITY EFFECT: Energy field at large scales =====
    // When sphere gets massive (scale > 8), reduce opacity for energy field look
    if (overlayMaterial) {
        if (sphereScale > 8) {
            // Gradually reduce opacity as scale increases
            const opacityFactor = Math.max(0.3, 1 - (sphereScale - 8) / 10);
            overlayMaterial.opacity = opacityFactor;
        } else {
            // Normal opacity at smaller scales
            overlayMaterial.opacity = 0.2;
        }
    }

    // Update attached light position
    if (pointLight) {
        pointLight.position.copy(sphere.parent.position);
    }

    // Update stats display
    document.getElementById('stat-x').textContent = spherePos.x.toFixed(1);
    document.getElementById('stat-y').textContent = spherePos.y.toFixed(1);
    document.getElementById('stat-z').textContent = spherePos.z.toFixed(1);
}

/**
 * Update performance stats
 */
function updateStats() {
    const now = Date.now();
    frameCount++;

    if (now - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
    }

    document.getElementById('stat-fps').textContent = fps;
    document.getElementById('stat-hands').textContent = handsDetected;
}

/**
 * Main render loop - continuous animation with hand detection
 */
function render() {
    if (!isRunning) return;

    animationId = requestAnimationFrame(render);

    const startTime = performance.now();

    // ===== DETECT HANDS USING MODERN HANDLANDMARKER API =====
    const video = document.getElementById('video');
    
    if (handLandmarker && video.readyState === video.HAVE_ENOUGH_DATA) {
        // Get current timestamp for VIDEO running mode
        const timestamp = performance.now();
        
        // Only process if enough time has passed (avoid redundant frames)
        if (timestamp !== lastVideoTime) {
            lastVideoTime = timestamp;
            
            try {
                // Detect hand landmarks from video frame
                const results = handLandmarker.detectForVideo(video, timestamp);
                
                // Process results
                processHandResults(results);
                
            } catch (error) {
                console.error('Hand detection error:', error);
            }
        }
    }

    // Update sphere with lerp smoothing
    updateSphere();

    // Update stats display
    updateStats();

    // Render 3D scene
    renderer.render(scene, camera);
}

// ==================== UI HANDLERS ====================

function setStatus(message, type = 'info') {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = 'status';
    if (type === 'error') status.classList.add('error');
    if (type === 'loading') status.classList.add('loading');
}

/**
 * Start hand tracking and camera
 */
async function startTracking() {
    try {
        setStatus('Initializing...', 'loading');
        console.log('=== STARTING HAND TRACKING ===');

        // Initialize Three.js
        initThreeJS();

        // Initialize MediaPipe
        await initMediaPipe();

        // Request camera with mobile-optimized constraints
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
                // Mobile optimization: request lower resolution on mobile devices
                ...(window.innerWidth < 768 && {
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                })
            }
        };

        console.log('Requesting camera (mobile-optimized)...');
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        const video = document.getElementById('video');
        video.srcObject = stream;

        // Wait for video to be ready
        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                console.log('✓ Camera ready:', {
                    width: video.videoWidth,
                    height: video.videoHeight
                });
                resolve();
            };
        });

        isRunning = true;
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').style.display = 'block';

        setStatus('Tracking active ✓', 'info');
        console.log('✓ Tracking started - watch the console for hand detection logs');

        // Start render/animation loop
        render();

    } catch (error) {
        console.error('✗ Start error:', error);
        setStatus(`Error: ${error.message}`, 'error');
        document.getElementById('start-btn').disabled = false;
    }
}

/**
 * Stop hand tracking and camera
 */
function stopTracking() {
    isRunning = false;
    console.log('=== STOPPING HAND TRACKING ===');

    // Stop camera
    const video = document.getElementById('video');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => {
            track.stop();
            console.log('✓ Camera track stopped');
        });
        video.srcObject = null;
    }

    // Stop animation
    if (animationId) {
        cancelAnimationFrame(animationId);
        console.log('✓ Animation loop stopped');
    }

    // Clean up Three.js
    if (renderer && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
        console.log('✓ Renderer cleaned up');
    }

    // Reset UI
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').style.display = 'none';
    setStatus('Camera stopped', 'info');

    console.log('✓ Tracking stopped');
}

// ==================== EVENT LISTENERS ====================

document.addEventListener('DOMContentLoaded', () => {
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    const deviceInfo = isMobile ? 'MOBILE' : 'DESKTOP';
    
    console.log('%c========== 3D HAND TRACKING (' + deviceInfo + ') ==========', 
        'color: #00ffff; font-size: 14px; font-weight: bold;');
    console.log('📱 Device:', isMobile ? 'Mobile (optimized performance)' : 'Desktop');
    console.log('📹 Click START CAMERA to begin');
    console.log('🎯 GPU acceleration enabled');
    console.log('✋ Follow your index finger - sphere tracks cursor position');
    console.log('✌️  Pinch thumb+index to shrink/grow sphere (spider-web effect)');
    console.log('🕸️  Watch the neon-green wireframe respond to pinch gestures');
    
    document.getElementById('start-btn').addEventListener('click', startTracking);
    document.getElementById('stop-btn').addEventListener('click', stopTracking);
    window.addEventListener('beforeunload', stopTracking);
});

// Export for debugging in console
window.app = {
    scene,
    camera,
    renderer,
    isRunning,
    spherePos,
    targetPos,
    sphereScale,
    targetScale,
    handLandmarker,
    CONFIG,
    // Helper functions
    getHandStats: () => ({
        handsDetected,
        lastConfidence,
        frameCounter,
        spherePos,
        targetPos,
        sphereScale,
        targetScale
    })
};

console.log('%c✓ App exported to window.app', 'color: #00ff00; font-weight: bold;');

