/* Three.js scene for deposition frames and calibrated REP5X CAD parts.
   The CAD loader requires a measured common home coordinate frame.  It never
   invents pivots or implies a machine model when calibration is absent. */
(function () {
  "use strict";
  const host = document.getElementById("rep5xScene");
  if (!host || !window.THREE) return;
  const T = window.THREE;
  const scene = new T.Scene();
  scene.background = new T.Color(0x0b1020);
  const camera = new T.PerspectiveCamera(45, 1, 0.01, 2000);
  camera.position.set(12, -16, 12);
  const renderer = new T.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.appendChild(renderer.domElement);
  const controls = T.OrbitControls ? new T.OrbitControls(camera, renderer.domElement) : null;
  if (controls) { controls.target.set(0, 0, 0); controls.update(); }
  scene.add(new T.AmbientLight(0xffffff, 0.75));
  const light = new T.DirectionalLight(0xffffff, 0.7);
  light.position.set(10, -10, 20); scene.add(light);
  const plate = new T.GridHelper(20, 40, 0x587096, 0x263852);
  plate.rotation.x = Math.PI / 2; scene.add(plate);
  const root = new T.Group(); scene.add(root);
  const carriage = new T.Group(); root.add(carriage);
  const baseC = new T.Group(); carriage.add(baseC);
  const jointC = new T.Group(); baseC.add(jointC);
  const baseB = new T.Group(); jointC.add(baseB);
  const jointB = new T.Group(); baseB.add(jointB);
  const original = new T.Group(); scene.add(original);
  const staticCad = new T.Group(); scene.add(staticCad);
  const path = new T.Group(); scene.add(path);
  const nozzleAxis = new T.Line(
    new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3(0, 0, -2)]),
    new T.LineBasicMaterial({color: 0xff8a8a}));
  scene.add(nozzleAxis);
  const nozzleTip = new T.Mesh(new T.SphereGeometry(0.13, 12, 8),
    new T.MeshStandardMaterial({color: 0xff8a8a})); scene.add(nozzleTip);
  let manifest = null, events = [], frames = [], voxels = [], deposition = null;
  let calibration = null, cadMeshes = [];
  const dummy = new T.Object3D();
  const assetStatus = document.getElementById("rep5xAssetStatus");
  const depStatus = document.getElementById("depositionStatus");
  const showMachine = document.getElementById("showRep5xMachine");
  const showDep = document.getElementById("showDeposition");
  const showAxis = document.getElementById("showNozzleAxis");
  const showPath = document.getElementById("showPhysicalPath");
  const showOriginal = document.getElementById("showPhysicalOriginal");

  function resize() {
    if (!host.clientWidth || !host.clientHeight) return;
    camera.aspect = host.clientWidth / host.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(host.clientWidth, host.clientHeight, false);
  }
  new ResizeObserver(resize).observe(host);
  function loop() {
    requestAnimationFrame(loop);
    if (!host.hidden) { resize(); if (controls) controls.update(); renderer.render(scene, camera); }
  }
  loop();
  function clearGroup(group) {
    while (group.children.length) {
      const item = group.children[0]; group.remove(item);
      if (item.geometry) item.geometry.dispose();
      if (item.material) item.material.dispose();
    }
  }
  function direction(b, c) {
    const br = b * Math.PI / 180, cr = c * Math.PI / 180;
    return new T.Vector3(-Math.sin(br)*Math.cos(cr),
      -Math.sin(br)*Math.sin(cr), -Math.cos(br));
  }
  function poseAt(time) {
    if (!events.length) return null;
    let event = events.find(e => time <= e.end_time + 1e-9) || events[events.length-1];
    const s = event.estimated_duration > 0 ? Math.min(1, Math.max(0,
      (time-event.start_time)/event.estimated_duration)) : 1;
    return {
      position: event.start_position.map((v, i) => v +
        (event.end_position[i]-v)*s),
      b: event.start_b + (event.end_b-event.start_b)*s,
      c: event.start_c + (event.end_c-event.start_c)*s,
      line: event.line, time_s: time
    };
  }
  function updateMachine(pose) {
    if (!pose) return;
    const tip = new T.Vector3(...pose.position), d = direction(pose.b, pose.c);
    nozzleTip.position.copy(tip);
    const axisPoints = [tip, tip.clone().addScaledVector(d, 2)];
    nozzleAxis.geometry.dispose();
    nozzleAxis.geometry = new T.BufferGeometry().setFromPoints(axisPoints);
    nozzleAxis.visible = showAxis.checked;
    if (!calibration) { carriage.visible = false; return; }
    carriage.visible = showMachine.checked;
    staticCad.visible = showMachine.checked;
    jointC.rotation.z = pose.c * Math.PI / 180;
    jointB.rotation.y = pose.b * Math.PI / 180;
    carriage.position.set(0, 0, 0);
    root.updateMatrixWorld(true);
    // The home CAD tip is transformed through B then C.  Translate the
    // carriage so the transformed tip equals the commanded RTCP XYZ.
    const tipLocal = new T.Vector3(...calibration.tipInB);
    const transformed = jointB.localToWorld(tipLocal.clone());
    carriage.position.add(tip.clone().sub(transformed));
    root.updateMatrixWorld(true);
    const check = jointB.localToWorld(tipLocal.clone());
    assetStatus.textContent = `CAD 보정: ${calibration.status || "UNVERIFIED"}; ` +
      `RTCP 오차 ${check.distanceTo(tip).toExponential(2)} mm`;
  }
  function setTime(time) {
    const pose = poseAt(time);
    updateMachine(pose);
    if (deposition) {
      let count = 0;
      for (const frame of frames) {
        if (frame.time_s > time + 1e-9) break;
        count += frame.added.length;
      }
      deposition.count = count;
      deposition.visible = showDep.checked;
      depStatus.textContent = `${manifest.backend}: ${count.toLocaleString()} voxel, ` +
        `${time.toFixed(3)} s · CFD ${manifest.cfd_applied ? "적용" : "미적용"}`;
    }
    path.visible = showPath.checked;
    original.visible = showOriginal.checked;
    return pose;
  }
  function loadPrediction(nextManifest, nextEvents, nextFrames) {
    if (nextManifest.backend !== "volco_oriented_extension" ||
        nextManifest.orientation_modeled !== true ||
        nextManifest.cfd_applied !== false)
      throw new Error("지원되지 않는 예측 manifest입니다. 물리 계산 상태를 확인하세요.");
    manifest = nextManifest; events = nextEvents; frames = nextFrames;
    if (deposition) {
      scene.remove(deposition); deposition.geometry.dispose();
      deposition.material.dispose();
    }
    voxels = frames.flatMap(frame => frame.added);
    const h = manifest.voxel_size_mm;
    deposition = new T.InstancedMesh(new T.BoxGeometry(h, h, h),
      new T.MeshStandardMaterial({color: 0x5ed7bf, roughness: 0.8}), voxels.length);
    voxels.forEach((ijk, index) => {
      dummy.position.set((ijk[0]+0.5)*h, (ijk[1]+0.5)*h,
        (ijk[2]+0.5)*h);
      dummy.updateMatrix(); deposition.setMatrixAt(index, dummy.matrix);
    });
    deposition.instanceMatrix.needsUpdate = true;
    scene.add(deposition);
    clearGroup(path);
    for (const event of events) {
      if (event.segment_length <= 0) continue;
      const geometry = new T.BufferGeometry().setFromPoints([
        new T.Vector3(...event.start_position), new T.Vector3(...event.end_position)]);
      path.add(new T.Line(geometry, new T.LineBasicMaterial({
        color: event.extruding ? 0xffc37a : 0x617084})));
    }
    setTime(events[events.length-1].end_time);
    const bounds = new T.Box3().setFromObject(deposition);
    if (!bounds.isEmpty()) {
      const center = bounds.getCenter(new T.Vector3());
      if (controls) { controls.target.copy(center); controls.update(); }
      camera.position.copy(center.clone().add(new T.Vector3(6, -8, 5)));
    }
  }
  function loadOriginal(stl) {
    clearGroup(original);
    const geometry = new T.BufferGeometry();
    const xyz = [];
    for (let t=0; t<stl.nTri; t++) {
      for (const index of [stl.fi[t], stl.fj[t], stl.fk[t]]) {
        xyz.push(stl.vx[index], stl.vy[index], stl.vz[index]);
      }
    }
    geometry.setAttribute("position", new T.Float32BufferAttribute(xyz, 3));
    geometry.computeVertexNormals();
    original.add(new T.Mesh(geometry, new T.MeshStandardMaterial({
      color: 0x6382af, transparent: true, opacity: 0.25, side: T.DoubleSide})));
  }
  function validatedCalibration(config) {
    if (config.units !== "mm" || config.coordinate_frame !== "common_home")
      throw new Error("CAD 단위 mm와 공통 home 좌표계를 명시하세요.");
    if (config.status === "UNCALIBRATED" || config.status === "PLACEHOLDER")
      throw new Error("보정 템플릿 상태입니다. 실측 행렬을 입력하고 상태를 갱신하세요.");
    for (const name of ["H_T_C", "C_T_B", "B_T_N"]) {
      const matrix = config.transforms?.[name];
      if (!Array.isArray(matrix) || matrix.length !== 16 ||
          !matrix.every(Number.isFinite))
        throw new Error(`${name}의 실측 4×4 행렬(행 우선)이 필요합니다.`);
      if (matrix.slice(12).some((value, i) =>
          Math.abs(value - [0, 0, 0, 1][i]) > 1e-9))
        throw new Error(`${name}의 마지막 행은 [0,0,0,1]이어야 합니다.`);
      const rotation = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]]
      ];
      const dot = (a, b) => a.reduce((sum, value, i) => sum + value*b[i], 0);
      const det = rotation[0][0]*(rotation[1][1]*rotation[2][2]-rotation[1][2]*rotation[2][1])
        - rotation[0][1]*(rotation[1][0]*rotation[2][2]-rotation[1][2]*rotation[2][0])
        + rotation[0][2]*(rotation[1][0]*rotation[2][1]-rotation[1][1]*rotation[2][0]);
      if (Math.abs(det-1) > 1e-5 || rotation.some((row, i) =>
          rotation.some((other, j) => Math.abs(dot(row, other)-(i===j ? 1 : 0)) > 1e-5)))
        throw new Error(`${name}은 스케일·반사가 없는 rigid transform이어야 합니다.`);
    }
    const axes = [["c_axis_local", [0, 0, 1]], ["b_axis_local", [0, 1, 0]]];
    for (const [name, expected] of axes) {
      if (!Array.isArray(config[name]) || config[name].length !== 3 ||
          !config[name].every((value, i) => Math.abs(value-expected[i]) < 1e-9))
        throw new Error(`${name}은 rep5x.py의 축 규약과 일치해야 합니다.`);
    }
    const hTc = matrixOf(config.transforms.H_T_C);
    const cTb = matrixOf(config.transforms.C_T_B);
    const bTn = matrixOf(config.transforms.B_T_N);
    const worldC = new T.Vector3(0, 0, 1).transformDirection(hTc);
    const worldB = new T.Vector3(0, 1, 0).transformDirection(hTc.clone().multiply(cTb));
    const homeNozzle = new T.Vector3(0, 0, -1).transformDirection(
      hTc.clone().multiply(cTb).multiply(bTn));
    if (worldC.distanceTo(new T.Vector3(0, 0, 1)) > 1e-6 ||
        worldB.distanceTo(new T.Vector3(0, 1, 0)) > 1e-6 ||
        homeNozzle.distanceTo(new T.Vector3(0, 0, -1)) > 1e-6)
      throw new Error("보정 CAD 축이 REP5X B/C 및 기본 노즐 방향과 일치하지 않습니다.");
    if (!config.parts || !config.parts.carriage || !config.parts.c_axis ||
        !config.parts.b_axis || !config.parts.hotend)
      throw new Error("carriage/C/B/hotend CAD 파일 매핑이 필요합니다.");
    return config;
  }
  function matrixOf(rows) { return new T.Matrix4().set(...rows); }
  function setBaseTransform(group, matrix) {
    matrix.decompose(group.position, group.quaternion, group.scale);
  }
  async function loadCad(config, files) {
    const nextCalibration = validatedCalibration(config);
    if (!T.STLLoader) throw new Error("Three.js STLLoader를 불러오지 못했습니다.");
    for (const fileName of Object.values(config.parts)) {
      if (!files.some(item => item.name === fileName))
        throw new Error(`CAD 파일 누락: ${fileName}`);
    }
    cadMeshes.forEach(mesh => {
      mesh.parent.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose();
    }); cadMeshes = [];
    const loader = new T.STLLoader();
    const hTc = matrixOf(config.transforms.H_T_C);
    const cTb = matrixOf(config.transforms.C_T_B);
    const bTn = matrixOf(config.transforms.B_T_N);
    setBaseTransform(baseC, hTc);
    setBaseTransform(baseB, cTb);
    nextCalibration.tipInB = new T.Vector3().setFromMatrixPosition(bTn).toArray();
    const homeB = hTc.clone().multiply(cTb);
    const groups = {frame: staticCad, carriage, c_axis: jointC,
      b_axis: jointB, hotend: jointB};
    for (const [name, fileName] of Object.entries(config.parts)) {
      const file = files.find(item => item.name === fileName);
      const geometry = loader.parse(await file.arrayBuffer());
      geometry.computeVertexNormals();
      // Every mesh shares home coordinates; translate its vertices into the
      // corresponding joint's local frame before parent rotation is applied.
      if (name === "c_axis") geometry.applyMatrix4(hTc.clone().invert());
      if (name === "b_axis" || name === "hotend")
        geometry.applyMatrix4(homeB.clone().invert());
      const mesh = new T.Mesh(geometry,
        new T.MeshStandardMaterial({color: 0x9ba9c5, roughness: 0.72}));
      (groups[name] || carriage).add(mesh); cadMeshes.push(mesh);
    }
    calibration = nextCalibration;
    assetStatus.textContent = "CAD 로드됨; RTCP 검증 대기";
  }
  function manualPose(x, y, z, b, c) {
    const pose = {position: [x, y, z], b, c, time_s: 0};
    updateMachine(pose);
    if (deposition) deposition.visible = showDep.checked;
    path.visible = showPath.checked;
    original.visible = showOriginal.checked;
    return direction(b, c).toArray();
  }
  for (const box of [showMachine, showDep, showAxis, showPath, showOriginal])
    box.addEventListener("change", () => {
      if (document.getElementById("motionSource").value === "manual") {
        const ids = ["manualX", "manualY", "manualZ", "manualB", "manualC"];
        manualPose(...ids.map(id => Number(document.getElementById(id).value)));
      } else setTime(window.rep5xScene.currentTime || 0);
    });
  window.rep5xScene = {
    get ready() { return !!manifest; },
    get duration() { return events.length ? events[events.length-1].end_time : 0; },
    get currentTime() { return this._time || 0; },
    setTime(time) { this._time = time; return setTime(time); },
    loadPrediction, loadOriginal, loadCad, manualPose,
    show() { host.hidden = false; resize(); },
    hide() { host.hidden = true; }
  };
})();
