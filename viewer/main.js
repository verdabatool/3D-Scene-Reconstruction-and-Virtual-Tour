import * as THREE from "https://unpkg.com/three@0.158.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.158.0/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "https://unpkg.com/three@0.158.0/examples/jsm/loaders/PLYLoader.js";

let scene, camera, renderer;
let currentIndex = 0;
let camerasData = [];
let animating = false;

init();
loadCameras();
loadPointCloud();
animate();

function init() {
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        0.01,
        1000
    );

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    window.addEventListener("click", onClick);
}

function loadPointCloud() {
    const loader = new PLYLoader();
    loader.load("../output/week3_after_ba.ply", geometry => {
        geometry.computeVertexNormals();
        const material = new THREE.PointsMaterial({
            size: 0.01,
            color: 0xffffff
        });
        const points = new THREE.Points(geometry, material);
        scene.add(points);
    });
}

async function loadCameras() {
    const response = await fetch("../output/week3_cameras_after_ba.json");
    camerasData = await response.json();

    setCameraPose(0);
}

function setCameraPose(idx) {
    const cam = camerasData[idx];
    currentIndex = idx;

    camera.position.set(
        cam.center[0],
        cam.center[1],
        cam.center[2]
    );
    const R = cam.R;
    const rot = new THREE.Matrix4().set(
        R[0][0], R[1][0], R[2][0], 0,
        R[0][1], R[1][1], R[2][1], 0,
        R[0][2], R[1][2], R[2][2], 0,
        0, 0, 0, 1
    );
    camera.quaternion.setFromRotationMatrix(rot);
    camera.lookAt(0, 0, 0);
}

function onClick() {
    if (animating || camerasData.length === 0) return;

    const next = (currentIndex + 1) % camerasData.length;
    animateToCamera(next);
}

function animateToCamera(targetIdx) {
    animating = true;

    const startPos = camera.position.clone();
    const startQuat = camera.quaternion.clone();

    const target = camerasData[targetIdx];
    const endPos = new THREE.Vector3(
        target.center[0],
        target.center[1],
        target.center[2]
    );

    const R = target.R;
    const endMat = new THREE.Matrix4().set(
        R[0][0], R[0][1], R[0][2], 0,
        R[1][0], R[1][1], R[1][2], 0,
        R[2][0], R[2][1], R[2][2], 0,
        0, 0, 0, 1
    );

    const endQuat = new THREE.Quaternion().setFromRotationMatrix(endMat);

    const duration = 1000;
    const startTime = performance.now();

    function step(time) {
        const t = Math.min((time - startTime) / duration, 1);

        camera.position.lerpVectors(startPos, endPos, t);
        THREE.Quaternion.slerp(startQuat, endQuat, camera.quaternion, t);

        if (t < 1) {
            requestAnimationFrame(step);
        } else {
            animating = false;
            currentIndex = targetIdx;
        }
    }

    requestAnimationFrame(step);
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
