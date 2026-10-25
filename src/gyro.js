// WebHID gyro aiming for the DualSense / DualSense Edge.
//
// Chromium only (Chrome/Edge), and only in a secure context (https or localhost).
// connect() must be called from a user gesture; autoConnect() re-opens a pad the
// user already allowed, without a prompt.
//
// Per ~250 Hz input report:
//   raw int16 -> factory calibration (feature report 0x05) -> deg/s
//   -> residual bias (factory bias + auto-calibration while still)
//   -> soft dead zone -> tiered smoothing
//   -> integrate over the controller's own sensor timestamp
//   -> accumulate degrees; the game calls consume() once per frame.
//
// Integrating at the sensor rate and draining per frame keeps the turn the same
// at any frame rate and loses no samples when a frame runs long.
//
// The same reports carry sticks and buttons, so if opening the device over
// WebHID removes it from navigator.getGamepads(), the controller still works.

const SONY = 0x054c;
const PRODUCT_IDS = [0x0ce6, 0x0df2]; // DualSense, DualSense Edge
const FALLBACK_DPS_PER_LSB = 1 / 16.4; // ~+-2000 dps across int16
const TS_TICK = 1 / 3e6;               // sensor timestamp unit is ~0.333 us
const STILL_DPS = 1.5;                 // below this on every axis = "still"
const DEADZONE_DPS = 0.25;
const SMOOTH_LO = 5, SMOOTH_HI = 25;   // fully smooth below LO, raw above HI
const HIST = 6;                        // ~24 ms of averaging at 250 Hz
const PAD_STALE_MS = 450;              // ignore HID buttons if reports stop

const isDS = (d) => d && d.vendorId === SONY && PRODUCT_IDS.includes(d.productId);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export class GyroInput {
  static get supported() {
    if (typeof navigator === 'undefined' || !navigator.hid) return false;
    if (typeof window !== 'undefined' && window.isSecureContext === false) return false;
    return true;
  }

  constructor() {
    this.supported = GyroInput.supported;
    this.enabled = false;
    this.sens = 1.0;          // multiplier, 0.25..4
    this.invertX = false;     // horizontal (yaw)
    this.invertY = false;     // vertical (pitch)
    this.connected = false;
    this.calibrating = false;
    this.state = this.supported ? 'idle' : 'unsupported';
    this.error = '';
    this.onchange = null;     // called on connect/disconnect/calibration changes

    this._dev = null;
    this._scale = [FALLBACK_DPS_PER_LSB, FALLBACK_DPS_PER_LSB, FALLBACK_DPS_PER_LSB];
    this._fbias = [0, 0, 0];  // factory gyro bias, raw counts
    this._bias = [0, 0, 0];   // residual bias, deg/s, learned at runtime
    this._hist = [];
    this._acc = { yaw: 0, pitch: 0 };
    this._lastTs = null;
    this._still = { t: 0, n: 0, s: [0, 0, 0] };
    this._manual = null;      // active manual recenter {t, n, s}
    this._calTimer = null;
    this._padSnapshot = null;
    this._padAt = 0;
    this._onReport = this._onReport.bind(this);
    this._listening = false;
  }

  // ---- persistence (same style as doodle_sens) ----
  loadSettings() {
    try {
      this.enabled = localStorage.getItem('doodle_gyro') === '1';
      const s = parseFloat(localStorage.getItem('doodle_gyro_sens'));
      if (Number.isFinite(s)) this.sens = clamp(s, 0.25, 4);
      this.invertY = localStorage.getItem('doodle_gyro_invy') === '1';
      this.invertX = localStorage.getItem('doodle_gyro_invx') === '1';
    } catch { /* storage can be unavailable */ }
  }

  saveSettings() {
    this.sens = clamp(this.sens, 0.25, 4);
    try {
      localStorage.setItem('doodle_gyro', this.enabled ? '1' : '0');
      localStorage.setItem('doodle_gyro_sens', String(this.sens));
      localStorage.setItem('doodle_gyro_invy', this.invertY ? '1' : '0');
      localStorage.setItem('doodle_gyro_invx', this.invertX ? '1' : '0');
    } catch { /* storage can be unavailable */ }
  }

  _emit() { if (this.onchange) { try { this.onchange(); } catch { /* ignore */ } } }

  _setState(state, error = '') {
    this.state = state;
    this.error = error;
    this._emit();
  }

  // ---- connection ----
  /** Shows the browser's device chooser. Must be called from a click/key handler. */
  async connect() {
    if (!this.supported) {
      this._setState('unsupported');
      throw new Error('WebHID is not available in this browser');
    }
    this._setState('connecting');
    let devices;
    try {
      devices = await navigator.hid.requestDevice({
        filters: PRODUCT_IDS.map((productId) => ({ vendorId: SONY, productId })),
      });
    } catch (err) {
      this._setState('error', err && err.message ? err.message : String(err));
      throw err;
    }
    if (!devices || !devices[0]) {
      this._setState(this.connected ? 'connected' : 'idle');
      return false;
    }
    await this._attach(devices[0]);
    return true;
  }

  /** Re-opens a previously permitted pad without a prompt. Safe to call at startup. */
  async autoConnect() {
    if (!this.supported || !this.enabled || this.connected) return false;
    if (!this._listening) {
      this._listening = true;
      navigator.hid.addEventListener('connect', (e) => {
        if (isDS(e.device) && !this._dev && this.enabled) this._attach(e.device).catch(() => {});
      });
      navigator.hid.addEventListener('disconnect', (e) => {
        if (e.device === this._dev) this._detach();
      });
    }
    try {
      const d = (await navigator.hid.getDevices()).find(isDS);
      if (d) { await this._attach(d); return true; }
    } catch { /* leave idle */ }
    return false;
  }

  async disconnect() {
    const d = this._dev;
    this._detach();
    try { if (d && d.opened) await d.close(); } catch { /* ignore */ }
  }

  async _attach(d) {
    if (this._dev === d && this.connected) return;
    if (this._dev && this._dev !== d) {
      try { this._dev.removeEventListener('inputreport', this._onReport); } catch { /* ignore */ }
      try { if (this._dev.opened) await this._dev.close(); } catch { /* ignore */ }
      this._dev = null;
    }
    this._setState('connecting');
    try {
      if (!d.opened) await d.open();
      await this._loadCalibration(d);
      d.addEventListener('inputreport', this._onReport);
      this._dev = d;
      this._lastTs = null;
      this._hist.length = 0;
      this._acc.yaw = this._acc.pitch = 0;
      this._bias = [0, 0, 0];
      this._resetStill();
      this._clearManual();
      this.connected = true;
      this._setState('connected');
    } catch (err) {
      try { if (d.opened) await d.close(); } catch { /* ignore */ }
      this._setState('error', err && err.message ? err.message : String(err));
      throw err;
    }
  }

  _detach() {
    if (this._dev) {
      try { this._dev.removeEventListener('inputreport', this._onReport); } catch { /* ignore */ }
    }
    this._dev = null;
    this.connected = false;
    this._acc.yaw = this._acc.pitch = 0;
    this._lastTs = null;
    this._hist.length = 0;
    this._clearManual();
    this._setState(this.supported ? 'idle' : 'unsupported');
  }

  // Feature report 0x05 is the factory IMU calibration. Reading it also switches
  // a Bluetooth pad from the short 0x01 report to full 0x31 reports with IMU data.
  //
  // Chromium's receiveFeatureReport may or may not include the report ID in the
  // returned DataView, so both layouts are attempted and the one that validates
  // wins. Field offsets follow Linux hid-playstation / SDL (data[1] = pitch bias).
  async _loadCalibration(d) {
    try {
      if (typeof d.receiveFeatureReport !== 'function') return false;
      const view = await d.receiveFeatureReport(0x05);
      return this._applyCalibration(view);
    } catch {
      return false; // keep the fallback scale; runtime bias still handles drift
    }
  }

  _applyCalibration(view) {
    if (!view || typeof view.getInt16 !== 'function' || view.byteLength < 21) return false;
    const parse = (base) => {
      if (view.byteLength < base + 23) return null;
      const i16 = (o) => view.getInt16(o, true);
      const bias = [i16(base + 1), i16(base + 3), i16(base + 5)];
      const speed2x = i16(base + 19) + i16(base + 21);
      const pairs = [[base + 7, base + 9], [base + 11, base + 13], [base + 15, base + 17]];
      const scale = pairs.map(([p, m]) => {
        const denom = i16(p) - i16(m);
        return denom ? speed2x / denom : 0;
      });
      // Plausibility checks mirror SDL's: a sane gyro scale is ~1/16.4 dps per count.
      if (bias.some((b) => Math.abs(b) > 1024)) return null;
      if (scale.some((s) => !(s > 0.02 && s < 0.2))) return null;
      return { bias, scale };
    };
    let parsed = null;
    // 0x05 is the report ID, which only appears if Chromium kept it in the data.
    if (view.getUint8(0) === 0x05) parsed = parse(0);
    if (!parsed) parsed = parse(-1);
    if (!parsed) parsed = parse(0);
    if (!parsed) return false;
    this._fbias = parsed.bias;
    this._scale = parsed.scale;
    return true;
  }

  // ---- calibration ----
  /** Hold the pad still for ~1 s. Look output pauses while it samples bias. */
  recenter() {
    if (!this.connected) return false;
    this._manual = { t: 0, n: 0, s: [0, 0, 0] };
    this.calibrating = true;
    this._hist.length = 0;
    this._acc.yaw = this._acc.pitch = 0;
    clearTimeout(this._calTimer);
    // Reports can stop (pad sleeps); finish with whatever was collected.
    this._calTimer = setTimeout(() => this._finishManual(), 1600);
    this._emit();
    return true;
  }

  _finishManual() {
    if (!this._manual) return;
    const m = this._manual;
    if (m.n > 0) this._bias = [m.s[0] / m.n, m.s[1] / m.n, m.s[2] / m.n];
    this._clearManual();
    this._emit();
  }

  _clearManual() {
    clearTimeout(this._calTimer);
    this._calTimer = null;
    this._manual = null;
    this.calibrating = false;
  }

  _resetStill() { this._still = { t: 0, n: 0, s: [0, 0, 0] }; }

  // ---- report handling ----
  _onReport(e) {
    const { reportId, data } = e;
    let layout = null;
    // USB 0x01: 63 data bytes (64 minus the report ID), IMU at 15.
    // Bluetooth 0x31: 77 bytes, one padding byte, IMU at 16.
    // WebHID's event.data never includes the report ID.
    if (reportId === 0x01 && data.byteLength >= 63) {
      layout = { gyro: 15, ts: 27, sticks: 0, trig: 4, b0: 7, b1: 8, b2: 9 };
    } else if (reportId === 0x31 && data.byteLength >= 42) {
      layout = { gyro: 16, ts: 28, sticks: 1, trig: 5, b0: 8, b1: 9, b2: 10 };
    } else {
      return; // 10-byte Bluetooth simple report: no IMU, no full pad
    }

    this._readPad(data, layout);

    if (data.byteLength < layout.ts + 4) return;
    const gx = data.getInt16(layout.gyro, true);      // pitch
    const gy = data.getInt16(layout.gyro + 2, true);  // yaw
    const gz = data.getInt16(layout.gyro + 4, true);  // roll
    const ts = data.getUint32(layout.ts, true);

    let dt = 0.004;
    if (this._lastTs !== null) {
      const ticks = (ts - this._lastTs) >>> 0; // unsigned handles the 32-bit wrap
      const d = ticks * TS_TICK;
      if (ticks === 0) dt = 0;
      else if (d > 0 && d < 0.05) dt = d;
      else dt = 0; // a long gap (tab hidden / pad slept): don't guess a huge turn
    }
    this._lastTs = ts;
    this._process(gx, gy, gz, dt);
  }

  _readPad(data, layout) {
    const bx = data.getUint8(layout.sticks);
    const by = data.getUint8(layout.sticks + 1);
    const rx = data.getUint8(layout.sticks + 2);
    const ry = data.getUint8(layout.sticks + 3);
    const l2 = data.getUint8(layout.trig) / 255;
    const r2 = data.getUint8(layout.trig + 1) / 255;
    const b0 = data.getUint8(layout.b0);
    const b1 = data.getUint8(layout.b1);
    const b2 = data.getUint8(layout.b2);
    const dpad = b0 & 0x0f;
    const buttons = Array.from({ length: 18 }, () => ({ pressed: false, value: 0 }));
    const set = (i, on, value = 1) => { buttons[i] = { pressed: !!on, value: on ? value : 0 }; };

    set(0, b0 & 0x20); // cross
    set(1, b0 & 0x40); // circle
    set(2, b0 & 0x10); // square
    set(3, b0 & 0x80); // triangle
    set(4, b1 & 0x01); // L1
    set(5, b1 & 0x02); // R1
    set(6, (b1 & 0x04) || l2 > 0.35, l2); // L2
    set(7, (b1 & 0x08) || r2 > 0.35, r2); // R2
    set(8, b1 & 0x10); // create
    set(9, b1 & 0x20); // options
    set(10, b1 & 0x40); // L3
    set(11, b1 & 0x80); // R3
    set(12, dpad === 0 || dpad === 1 || dpad === 7); // up
    set(13, dpad === 3 || dpad === 4 || dpad === 5); // down
    set(14, dpad === 5 || dpad === 6 || dpad === 7); // left
    set(15, dpad === 1 || dpad === 2 || dpad === 3); // right
    set(16, b2 & 0x01); // PS
    set(17, b2 & 0x02); // touchpad click

    const axis = (v) => clamp((v - 128) / 127, -1, 1);
    this._padSnapshot = { axes: [axis(bx), axis(by), axis(rx), axis(ry)], buttons };
    this._padAt = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  }

  /** Sticks/buttons from the last HID report, or null if they have gone stale. */
  get pad() {
    if (!this.connected || !this._padSnapshot) return null;
    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    return now - this._padAt < PAD_STALE_MS ? this._padSnapshot : null;
  }

  /** Exposed for tests: consume one IMU sample. */
  _process(gx, gy, gz, dt) {
    const f = this._fbias, s = this._scale, b = this._bias;
    const u = [
      (gx - f[0]) * s[0], // pitch, deg/s
      (gy - f[1]) * s[1], // yaw
      (gz - f[2]) * s[2], // roll
    ];

    // Manual recenter: average ~1 s of still samples and adopt the result.
    if (this._manual) {
      const m = this._manual;
      m.t += dt; m.n++;
      m.s[0] += u[0]; m.s[1] += u[1]; m.s[2] += u[2];
      if (m.t >= 1) this._finishManual();
      return;
    }

    const pitch = u[0] - b[0], yaw = u[1] - b[1], roll = u[2] - b[2];

    // Continuous auto-calibration while the pad sits still.
    if (Math.abs(pitch) < STILL_DPS && Math.abs(yaw) < STILL_DPS && Math.abs(roll) < STILL_DPS) {
      const st = this._still;
      st.t += dt; st.n++;
      st.s[0] += u[0]; st.s[1] += u[1]; st.s[2] += u[2];
      if (st.t >= 0.5) {
        for (let i = 0; i < 3; i++) b[i] += (st.s[i] / st.n - b[i]) * 0.3;
        this._resetStill();
      }
    } else if (this._still.n) {
      this._resetStill();
    }

    if (!this.enabled) return;

    // Soft dead zone: kill idle jitter without a hard edge.
    const sp = Math.hypot(yaw, pitch);
    const k = sp <= DEADZONE_DPS ? 0 : sp >= DEADZONE_DPS * 3 ? 1 : (sp - DEADZONE_DPS) / (DEADZONE_DPS * 2);
    const cy = yaw * k, cp = pitch * k;

    // Tiered smoothing: average slow motion, pass fast flicks through untouched.
    const h = this._hist;
    h.push(cy, cp);
    if (h.length > HIST * 2) h.splice(0, 2);
    let ay = 0, ap = 0;
    for (let i = 0; i < h.length; i += 2) { ay += h[i]; ap += h[i + 1]; }
    ay /= h.length / 2; ap /= h.length / 2;
    const w = clamp((sp - SMOOTH_LO) / (SMOOTH_HI - SMOOTH_LO), 0, 1);
    const oy = ay + (cy - ay) * w, op = ap + (cp - ap) * w;

    this._acc.yaw = clamp(this._acc.yaw + oy * dt, -90, 90);
    this._acc.pitch = clamp(this._acc.pitch + op * dt, -90, 90);
  }

  /**
   * Call once per frame. Returns degrees turned since the last call:
   *   x > 0 = look right, y > 0 = look up.
   * Sensitivity and inversion are applied here so UI changes take effect now.
   *
   * With +X right, +Y up, +Z toward the player (SDL's controller frame), turning
   * right gives a negative yaw rate, which is also a negative look.x in this
   * game. Pitch up is positive. The invert toggles cover other grips.
   */
  consume() {
    const a = this._acc;
    const x = a.yaw * this.sens * (this.invertX ? -1 : 1);
    const y = a.pitch * this.sens * (this.invertY ? -1 : 1);
    a.yaw = a.pitch = 0;
    if (!this.enabled) return { x: 0, y: 0 };
    return { x, y };
  }

  /** Drop accumulated motion when the page was hidden or the pad reconnects. */
  reset() {
    this._acc.yaw = this._acc.pitch = 0;
    this._lastTs = null;
    this._hist.length = 0;
    this._resetStill();
    if (this._manual) {
      this._clearManual();
      this._emit();
    }
  }
}
