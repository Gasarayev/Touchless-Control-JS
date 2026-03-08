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
    lerpFactor: 0.1,
    sphereRadius: 25,
    depthScale: 150,
    scaleMultiplier: 45,       // Scale sphere size based on hand openness
    particleCount: 500,        // Number of falling particle dots
    fullyOpenThreshold: 0.35,  // Hand fully open threshold (lower = easier to trigger)
    emojiSpriteCount: 100      // Number of emoji sprites
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

// Particle system
let heartParticles = null;
let emojiGroup = null;           // Group of emoji sprites
let emojiSprites = [];           // Array to track individual sprite data
let congratulationsText = null;
let spherePos = { x: 0, y: 0, z: 0 };

// Opacity control for smooth transitions
let emojiOpacity = 0;
let textOpacity = 0;
let targetEmojiOpacity = 0;
let targetTextOpacity = 0;

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

/**
 * Create a canvas texture with an emoji drawn on it
 */
function createEmojiTexture(emoji) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');

    // Clear background
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fillRect(0, 0, 128, 128);

    // Draw emoji
    ctx.font = 'bold 100px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(emoji, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearFilter;
    return texture;
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
    // Dynamically adjust camera Z based on screen width (mobile: further back)
    const cameraPosZ = width < 768 ? 500 : 350;
    camera.position.set(0, 0, cameraPosZ);
    camera.lookAt(0, 0, 0);

    // Create renderer with full viewport
    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        precision: 'highp'
    });
    
    // Set size to match viewport exactly
    renderer.autoClear = true;
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio); // Use full device pixel ratio for sharp graphics
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

    // Create particle system for falling hearts/rain
    const particleGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(CONFIG.particleCount * 3);

    for (let i = 0; i < CONFIG.particleCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 10;
    }
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const pMaterial = new THREE.PointsMaterial({
        color: 0xff0000,
        size: 0.1,
        transparent: true,
        opacity: 0
    });

    heartParticles = new THREE.Points(particleGeometry, pMaterial);
    scene.add(heartParticles);

    // Create emoji sprites for celebration (8 March)
    const emojis = ['❤️', '💖', '💝', '💓', '💘'];
    emojiGroup = new THREE.Group();
    emojiSprites = [];

    // Create 100 emoji sprites scattered across screen
    for (let i = 0; i < CONFIG.emojiSpriteCount; i++) {
        // Pick random emoji
        const emoji = emojis[Math.floor(Math.random() * emojis.length)];
        const texture = createEmojiTexture(emoji);

        // Create sprite with emoji texture
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            opacity: 0,
            depthWrite: false,
            depthTest: false  // Always visible, don't test depth
        });

        const sprite = new THREE.Sprite(spriteMaterial);

        // Random position across full screen
        sprite.position.x = (Math.random() - 0.5) * 400;
        sprite.position.y = (Math.random() - 0.5) * 300;
        sprite.position.z = Math.random() * 200;  // Range: 0 to 200 (in front of camera at Z=350/500)

        // Responsive scale based on viewport size
        const emojiScale = Math.min(window.innerWidth, window.innerHeight) * 0.05;
        sprite.scale.set(emojiScale, emojiScale, 1);

        emojiGroup.add(sprite);
        emojiSprites.push({
            sprite: sprite,
            velocityY: -0.5 - Math.random() * 1,  // Falling speed
            velocityX: (Math.random() - 0.5) * 0.5,
            rotation: 0,
            rotationSpeed: (Math.random() - 0.5) * 0.05
        });
    }

    emojiGroup.position.set(0, 0, 0);
    emojiGroup.visible = false;
    scene.add(emojiGroup);

    // Create HTML-based celebration text overlay
    const textOverlay = document.createElement('div');
    textOverlay.id = 'celebration-text';
    textOverlay.textContent = 'Bayramınız Mübarək!💖';
    document.body.appendChild(textOverlay);
    congratulationsText = textOverlay;

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
        color: 0x00CCFF,           // Light blue base
        emissive: 0x00CCFF,        // Light blue glow
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
        color: 0x00CCFF,           // Light blue overlay
        emissive: 0x00CCFF,
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
    
    // Update camera aspect ratio
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    
    // Dynamically adjust camera Z based on screen width (mobile: further back)
    const cameraPosZ = width < 768 ? 500 : 350;
    camera.position.z = cameraPosZ;
    
    // Update renderer size and pixel ratio
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Rescale emoji sprites to match new viewport
    const emojiScale = Math.min(width, height) * 0.05;
    for (let spriteData of emojiSprites) {
        spriteData.sprite.scale.set(emojiScale, emojiScale, 1);
    }
    
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
    // This function is no longer used - prediction is done in the predict() function instead
}

// ==================== ANIMATION LOOP ====================

/**
 * Main prediction/render loop - continuous animation with hand detection
 */
function predict() {
    if (!isRunning) return;

    animationId = requestAnimationFrame(predict);

    const video = document.getElementById('video');
    const nowInMs = Date.now();

    if (handLandmarker && video.readyState === video.HAVE_ENOUGH_DATA) {
        const results = handLandmarker.detectForVideo(video, nowInMs);

        if (results.landmarks && results.landmarks.length > 0) {
            const thumb = results.landmarks[0][4];   // Thumb tip (landmark 4)
            const index = results.landmarks[0][8];   // Index finger tip (landmark 8)

            // Calculate distance between thumb and index finger
            const dx = thumb.x - index.x;
            const dy = thumb.y - index.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            // Map coordinates to 3D world space
            const targetX = (index.x - 0.5) * -12;
            const targetY = (index.y - 0.5) * -10;

            // ===== STATE 1: HAND CLOSED OR PARTIALLY OPEN (distance < threshold) =====
            if (distance < CONFIG.fullyOpenThreshold) {
                // Show sphere, hide celebration effects
                sphere.visible = true;
                targetEmojiOpacity = 0;
                targetTextOpacity = 0;

                // Scale sphere based on hand openness (distance * multiplier)
                const targetScale = distance * CONFIG.scaleMultiplier;
                sphere.parent.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);

                // Move sphere to follow index finger
                sphere.parent.position.x += (targetX - sphere.parent.position.x) * 0.2;
                sphere.parent.position.y += (targetY - sphere.parent.position.y) * 0.2;

                handsDetected = 1;
            }
            // ===== STATE 2: HAND FULLY OPEN (distance >= threshold) =====
            else {
                // Hide sphere, show celebration effects
                sphere.visible = false;
                emojiGroup.visible = true;  // Explicitly show emoji group
                targetEmojiOpacity = 1.0;
                targetTextOpacity = 1.0;

                // Get viewport dimensions for responsive boundaries
                const vh = window.innerHeight / 2;  // Half-height for centered camera
                const vw = window.innerWidth / 2;   // Half-width for centered viewport

                // Update emoji sprites: falling motion and rotation
                for (let spriteData of emojiSprites) {
                    const sprite = spriteData.sprite;

                    // Falling/floating motion
                    sprite.position.y += spriteData.velocityY;
                    sprite.position.x += spriteData.velocityX;

                    // Rotation (use rotation.z for sprites)
                    spriteData.rotation += spriteData.rotationSpeed;
                    sprite.rotation.z = spriteData.rotation;

                    // Reset sprite at top when it falls too low (based on viewport height)
                    if (sprite.position.y < -vh) {
                        sprite.position.y = vh;
                        sprite.position.x = (Math.random() - 0.5) * (vw * 2);
                    }

                    // Reset sprite if it drifts outside horizontal bounds
                    if (sprite.position.x < -vw * 1.5) {
                        sprite.position.x = vw * 1.5;
                    } else if (sprite.position.x > vw * 1.5) {
                        sprite.position.x = -vw * 1.5;
                    }
                }

                handsDetected = 1;
                if (distance > CONFIG.fullyOpenThreshold + 0.05) {
                    console.log('🎉 8 MARCH CELEBRATION! Hand fully open - distance:', distance.toFixed(3));
                }
            }
        } else {
            handsDetected = 0;
            targetEmojiOpacity = 0;
            targetTextOpacity = 0;
        }
    }

    // ===== SMOOTH LERP TRANSITIONS FOR OPACITY =====
    // Lerp emoji opacity
    emojiOpacity = lerp(emojiOpacity, targetEmojiOpacity, CONFIG.lerpFactor);
    if (emojiGroup) {
        emojiGroup.visible = emojiOpacity > 0.01;
        // Update opacity for all emoji sprites
        for (let spriteData of emojiSprites) {
            spriteData.sprite.material.opacity = emojiOpacity;
        }
    }

    // Lerp text opacity and toggle display
    textOpacity = lerp(textOpacity, targetTextOpacity, CONFIG.lerpFactor);
    if (congratulationsText) {
        congratulationsText.style.opacity = textOpacity.toString();
        congratulationsText.style.display = textOpacity > 0.01 ? 'block' : 'none';
    }

    // Update stats
    const now = Date.now();
    frameCount++;
    if (now - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
    }

    // Stats box removed - logo now displays in top-left

    // Update light to follow sphere
    if (pointLight && sphere && sphere.parent) {
        pointLight.position.copy(sphere.parent.position);
    }

    // Render scene
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

        setStatus('İzləmə aktivdir ✓', 'info');
        console.log('✓ Tracking started - watch the console for hand detection logs');

        // Start prediction/animation loop
        predict();

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

    // Clean up HTML text overlay
    if (congratulationsText && congratulationsText.parentNode) {
        congratulationsText.parentNode.removeChild(congratulationsText);
        congratulationsText = null;
        console.log('✓ Celebration text removed');
    }

    // Reset opacity states
    emojiOpacity = 0;
    textOpacity = 0;
    targetEmojiOpacity = 0;
    targetTextOpacity = 0;

    // Reset UI
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').style.display = 'none';
    setStatus('Kamera dayandırıldı', 'info');

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
    handLandmarker,
    CONFIG,
    // Helper functions
    getHandStats: () => ({
        handsDetected,
        lastConfidence,
        frameCounter,
        spherePos
    })
};

console.log('%c✓ App exported to window.app', 'color: #00ff00; font-weight: bold;');

