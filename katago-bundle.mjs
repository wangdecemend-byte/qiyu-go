// katago-engine/katago-entry.ts
import * as tf2 from "@tensorflow/tfjs";

// katago-engine/binModelParser.ts
var KataGoBinModelParser = class {
  data;
  idx = 0;
  decoder = new TextDecoder("utf-8");
  constructor(data) {
    this.data = data;
  }
  skipWhitespace() {
    while (this.idx < this.data.length) {
      const b = this.data[this.idx];
      if (b === 32 || b === 10 || b === 13 || b === 9) {
        this.idx++;
        continue;
      }
      break;
    }
  }
  readToken() {
    this.skipWhitespace();
    const start = this.idx;
    while (this.idx < this.data.length) {
      const b = this.data[this.idx];
      if (b === 32 || b === 10 || b === 13 || b === 9) break;
      this.idx++;
    }
    if (this.idx <= start) {
      throw new Error("Unexpected EOF while reading token");
    }
    return this.decoder.decode(this.data.subarray(start, this.idx));
  }
  readInt() {
    const tok = this.readToken();
    const value = Number.parseInt(tok, 10);
    if (!Number.isFinite(value)) throw new Error(`Invalid int token: ${tok}`);
    return value;
  }
  readFloatAscii() {
    const tok = this.readToken();
    const value = Number.parseFloat(tok);
    if (!Number.isFinite(value)) throw new Error(`Invalid float token: ${tok}`);
    return value;
  }
  readBinaryFloats(count) {
    this.skipWhitespace();
    const marker = this.data.subarray(this.idx, this.idx + 5);
    if (marker.length !== 5 || marker[0] !== 64 || // @
    marker[1] !== 66 || // B
    marker[2] !== 73 || // I
    marker[3] !== 78 || // N
    marker[4] !== 64) {
      throw new Error("Expected @BIN@ marker");
    }
    this.idx += 5;
    const byteLen = count * 4;
    const absStart = this.data.byteOffset + this.idx;
    const absEnd = absStart + byteLen;
    if (absEnd > this.data.buffer.byteLength) throw new Error("Unexpected EOF while reading binary floats");
    const buf = this.data.buffer.slice(absStart, absEnd);
    this.idx += byteLen;
    this.skipWhitespace();
    return new Float32Array(buf);
  }
};
function parseBatchNormV8(p) {
  p.readToken();
  const channels = p.readInt();
  const epsilon = p.readFloatAscii();
  const hasScale = p.readInt() !== 0;
  const hasBias = p.readInt() !== 0;
  const mean2 = p.readBinaryFloats(channels);
  const variance = p.readBinaryFloats(channels);
  const scale = hasScale ? p.readBinaryFloats(channels) : new Float32Array(channels).fill(1);
  const bias = hasBias ? p.readBinaryFloats(channels) : new Float32Array(channels).fill(0);
  const mergedScale = new Float32Array(channels);
  const mergedBias = new Float32Array(channels);
  for (let i = 0; i < channels; i++) {
    const ms = scale[i] / Math.sqrt(variance[i] + epsilon);
    mergedScale[i] = ms;
    mergedBias[i] = bias[i] - ms * mean2[i];
  }
  return { channels, mergedScale, mergedBias };
}
function parseActivationKind(p, modelVersion) {
  p.readToken();
  if (modelVersion < 11) return "relu";
  const kindTok = p.readToken();
  if (kindTok === "ACTIVATION_IDENTITY") return "identity";
  if (kindTok === "ACTIVATION_RELU") return "relu";
  if (kindTok === "ACTIVATION_MISH") return "mish";
  throw new Error(`Unsupported activation kind token: ${kindTok}`);
}
function parseConv2d(p) {
  const name = p.readToken();
  const kernelY = p.readInt();
  const kernelX = p.readInt();
  const inChannels = p.readInt();
  const outChannels = p.readInt();
  const dilationY = p.readInt();
  const dilationX = p.readInt();
  const weights = p.readBinaryFloats(kernelY * kernelX * inChannels * outChannels);
  return { name, kernelY, kernelX, inChannels, outChannels, dilationY, dilationX, weights };
}
function parseMatMul(p) {
  const name = p.readToken();
  const inChannels = p.readInt();
  const outChannels = p.readInt();
  const weights = p.readBinaryFloats(inChannels * outChannels);
  return { name, inChannels, outChannels, weights };
}
function parseMatBias(p) {
  const name = p.readToken();
  const channels = p.readInt();
  const weights = p.readBinaryFloats(channels);
  return { name, channels, weights };
}

// katago-engine/loadModelV8.ts
function parseKataGoModelV8(data) {
  const p = new KataGoBinModelParser(data);
  const modelName2 = p.readToken();
  const modelVersion = p.readInt();
  if (modelVersion < 8 || modelVersion > 16) {
    throw new Error(`Unsupported modelVersion ${modelVersion}, supported 8..16`);
  }
  const numInputChannels = p.readInt();
  const numInputGlobalChannels = p.readInt();
  const postProcessParams = modelVersion >= 13 ? {
    tdScoreMultiplier: p.readFloatAscii(),
    scoreMeanMultiplier: p.readFloatAscii(),
    scoreStdevMultiplier: p.readFloatAscii(),
    leadMultiplier: p.readFloatAscii(),
    varianceTimeMultiplier: p.readFloatAscii(),
    shorttermValueErrorMultiplier: p.readFloatAscii(),
    shorttermScoreErrorMultiplier: p.readFloatAscii(),
    outputScaleMultiplier: 1
  } : {
    // Defaults for older models (ModelPostProcessParams).
    tdScoreMultiplier: 20,
    scoreMeanMultiplier: 20,
    scoreStdevMultiplier: 20,
    leadMultiplier: 20,
    varianceTimeMultiplier: 40,
    shorttermValueErrorMultiplier: 0.25,
    shorttermScoreErrorMultiplier: 30,
    outputScaleMultiplier: 1
  };
  const metaEncoderVersion = modelVersion >= 15 ? p.readInt() : 0;
  if (modelVersion >= 15) {
    for (let i = 0; i < 7; i++) p.readInt();
  }
  if (metaEncoderVersion !== 0) {
    throw new Error(`Unsupported metaEncoderVersion ${metaEncoderVersion}`);
  }
  p.readToken();
  const numBlocks = p.readInt();
  const trunkNumChannels = p.readInt();
  const midNumChannels = p.readInt();
  const regularNumChannels = p.readInt();
  p.readInt();
  const gpoolNumChannels = p.readInt();
  if (modelVersion >= 15) {
    for (let i = 0; i < 6; i++) p.readInt();
  }
  const conv1 = parseConv2d(p);
  const ginput = parseMatMul(p);
  function parseResidualBlock() {
    const kindTok = p.readToken();
    if (kindTok === "ordinary_block") {
      p.readToken();
      const preBN = parseBatchNormV8(p);
      const preActivation = parseActivationKind(p, modelVersion);
      const w1 = parseConv2d(p);
      const midBN = parseBatchNormV8(p);
      const midActivation = parseActivationKind(p, modelVersion);
      const w2 = parseConv2d(p);
      return { kind: "ordinary", preBN, preActivation, w1, midBN, midActivation, w2 };
    }
    if (kindTok === "gpool_block") {
      p.readToken();
      const preBN = parseBatchNormV8(p);
      const preActivation = parseActivationKind(p, modelVersion);
      const w1a = parseConv2d(p);
      const w1b = parseConv2d(p);
      const gpoolBN = parseBatchNormV8(p);
      const gpoolActivation = parseActivationKind(p, modelVersion);
      const w1r = parseMatMul(p);
      const midBN = parseBatchNormV8(p);
      const midActivation = parseActivationKind(p, modelVersion);
      const w2 = parseConv2d(p);
      return { kind: "gpool", preBN, preActivation, w1a, w1b, gpoolBN, gpoolActivation, w1r, midBN, midActivation, w2 };
    }
    if (kindTok === "nested_bottleneck_block") {
      p.readToken();
      const numInnerBlocks = p.readInt();
      const preBN = parseBatchNormV8(p);
      const preActivation = parseActivationKind(p, modelVersion);
      const preConv = parseConv2d(p);
      const blocks2 = [];
      for (let i = 0; i < numInnerBlocks; i++) blocks2.push(parseResidualBlock());
      const postBN = parseBatchNormV8(p);
      const postActivation = parseActivationKind(p, modelVersion);
      const postConv = parseConv2d(p);
      return { kind: "nested_bottleneck", numBlocks: numInnerBlocks, preBN, preActivation, preConv, blocks: blocks2, postBN, postActivation, postConv };
    }
    throw new Error(`Unsupported trunk block kind ${kindTok}`);
  }
  const blocks = [];
  for (let i = 0; i < numBlocks; i++) blocks.push(parseResidualBlock());
  const tipBN = parseBatchNormV8(p);
  const tipActivation = parseActivationKind(p, modelVersion);
  p.readToken();
  const p1 = parseConv2d(p);
  const g1 = parseConv2d(p);
  const g1BN = parseBatchNormV8(p);
  const g1Activation = parseActivationKind(p, modelVersion);
  const gpoolToBias = parseMatMul(p);
  const p1BN = parseBatchNormV8(p);
  const p1Activation = parseActivationKind(p, modelVersion);
  const p2 = parseConv2d(p);
  const passMul = parseMatMul(p);
  const passBias = modelVersion >= 15 ? parseMatBias(p) : void 0;
  const passActivation = modelVersion >= 15 ? parseActivationKind(p, modelVersion) : void 0;
  const passMul2 = modelVersion >= 15 ? parseMatMul(p) : void 0;
  p.readToken();
  const v1 = parseConv2d(p);
  const v1BN = parseBatchNormV8(p);
  const v1Activation = parseActivationKind(p, modelVersion);
  const v2 = parseMatMul(p);
  const v2Bias = parseMatBias(p);
  const v2Activation = parseActivationKind(p, modelVersion);
  const v3 = parseMatMul(p);
  const v3Bias = parseMatBias(p);
  const sv3 = parseMatMul(p);
  const sv3Bias = parseMatBias(p);
  const ownership = parseConv2d(p);
  return {
    modelName: modelName2,
    modelVersion,
    numInputChannels,
    numInputGlobalChannels,
    metaEncoderVersion,
    postProcessParams,
    policyOutChannels: p2.outChannels,
    scoreValueChannels: sv3.outChannels,
    trunk: {
      numBlocks,
      trunkNumChannels,
      midNumChannels,
      regularNumChannels,
      gpoolNumChannels,
      conv1,
      ginput,
      blocks,
      tipBN,
      tipActivation
    },
    policy: {
      p1,
      g1,
      g1BN,
      g1Activation,
      gpoolToBias,
      p1BN,
      p1Activation,
      p2,
      passMul,
      passBias,
      passActivation,
      passMul2
    },
    value: {
      v1,
      v1BN,
      v1Activation,
      v2,
      v2Bias,
      v2Activation,
      v3,
      v3Bias,
      sv3,
      sv3Bias,
      ownership
    }
  };
}

// katago-engine/modelV8.ts
import * as tf from "@tensorflow/tfjs";
function makeBn(bn) {
  const scale = tf.tensor4d(bn.mergedScale, [1, 1, 1, bn.channels]);
  const bias = tf.tensor4d(bn.mergedBias, [1, 1, 1, bn.channels]);
  return { scale, bias };
}
function makeConv(conv) {
  const filter = tf.tensor4d(conv.weights, [conv.kernelY, conv.kernelX, conv.inChannels, conv.outChannels]);
  return {
    kernelY: conv.kernelY,
    kernelX: conv.kernelX,
    inChannels: conv.inChannels,
    outChannels: conv.outChannels,
    dilationY: conv.dilationY,
    dilationX: conv.dilationX,
    filter
  };
}
function makeMatMul(mm) {
  const w = tf.tensor2d(mm.weights, [mm.inChannels, mm.outChannels]);
  return { inChannels: mm.inChannels, outChannels: mm.outChannels, w };
}
function makeMatBias(bias) {
  const b = tf.tensor2d(bias.weights, [1, bias.channels]);
  return { channels: bias.channels, b };
}
function applyActivation4D(x, kind) {
  if (kind === "identity") return x;
  if (kind === "relu") return tf.relu(x);
  return tf.mul(x, tf.tanh(tf.softplus(x)));
}
function applyActivation2D(x, kind) {
  if (kind === "identity") return x;
  if (kind === "relu") return tf.relu(x);
  return tf.mul(x, tf.tanh(tf.softplus(x)));
}
function bnAct(x, bn, activation) {
  const y = tf.add(tf.mul(x, bn.scale), bn.bias);
  return applyActivation4D(y, activation);
}
function conv2d2(x, conv) {
  return tf.conv2d(x, conv.filter, 1, "same", "NHWC", [conv.dilationY, conv.dilationX]);
}
function poolRowsGPool(x) {
  const boardSize = x.shape[1] ?? 19;
  const factor = (boardSize - 14) * 0.1;
  const mean2 = tf.mean(x, [1, 2]);
  const max2 = tf.max(x, [1, 2]);
  return tf.concat([mean2, mean2.mul(factor), max2], 1);
}
function poolRowsValueHead(x) {
  const boardSize = x.shape[1] ?? 19;
  const base = boardSize - 14;
  const factor1 = base * 0.1;
  const factor2 = base * base * 0.01 - 0.1;
  const mean2 = tf.mean(x, [1, 2]);
  return tf.concat([mean2, mean2.mul(factor1), mean2.mul(factor2)], 1);
}
var KataGoModelV8Tf = class {
  modelName;
  modelVersion;
  postProcessParams;
  policyOutChannels;
  scoreValueChannels;
  trunkConv1;
  trunkGInput;
  trunkBlocks;
  trunkTipBN;
  trunkTipActivation;
  p1;
  g1;
  g1BN;
  g1Activation;
  gpoolToBias;
  p1BN;
  p1Activation;
  p2;
  passMul;
  passBias;
  passActivation;
  passMul2;
  v1;
  v1BN;
  v1Activation;
  v2;
  v2Bias;
  v2Activation;
  v3;
  v3Bias;
  sv3;
  sv3Bias;
  ownership;
  constructor(parsed) {
    this.modelName = parsed.modelName;
    this.modelVersion = parsed.modelVersion;
    this.postProcessParams = parsed.postProcessParams;
    this.policyOutChannels = parsed.policyOutChannels;
    this.scoreValueChannels = parsed.scoreValueChannels;
    this.trunkConv1 = makeConv(parsed.trunk.conv1);
    this.trunkGInput = makeMatMul(parsed.trunk.ginput);
    const toTfBlock = (b) => {
      if (b.kind === "ordinary") {
        return {
          kind: "ordinary",
          preBN: makeBn(b.preBN),
          preActivation: b.preActivation,
          w1: makeConv(b.w1),
          midBN: makeBn(b.midBN),
          midActivation: b.midActivation,
          w2: makeConv(b.w2)
        };
      }
      if (b.kind === "gpool") {
        return {
          kind: "gpool",
          preBN: makeBn(b.preBN),
          preActivation: b.preActivation,
          w1a: makeConv(b.w1a),
          w1b: makeConv(b.w1b),
          gpoolBN: makeBn(b.gpoolBN),
          gpoolActivation: b.gpoolActivation,
          w1r: makeMatMul(b.w1r),
          midBN: makeBn(b.midBN),
          midActivation: b.midActivation,
          w2: makeConv(b.w2)
        };
      }
      return {
        kind: "nested_bottleneck",
        numBlocks: b.numBlocks,
        preBN: makeBn(b.preBN),
        preActivation: b.preActivation,
        preConv: makeConv(b.preConv),
        blocks: b.blocks.map(toTfBlock),
        postBN: makeBn(b.postBN),
        postActivation: b.postActivation,
        postConv: makeConv(b.postConv)
      };
    };
    this.trunkBlocks = parsed.trunk.blocks.map(toTfBlock);
    this.trunkTipBN = makeBn(parsed.trunk.tipBN);
    this.trunkTipActivation = parsed.trunk.tipActivation;
    this.p1 = makeConv(parsed.policy.p1);
    this.g1 = makeConv(parsed.policy.g1);
    this.g1BN = makeBn(parsed.policy.g1BN);
    this.g1Activation = parsed.policy.g1Activation;
    this.gpoolToBias = makeMatMul(parsed.policy.gpoolToBias);
    this.p1BN = makeBn(parsed.policy.p1BN);
    this.p1Activation = parsed.policy.p1Activation;
    this.p2 = makeConv(parsed.policy.p2);
    this.passMul = makeMatMul(parsed.policy.passMul);
    this.passBias = parsed.policy.passBias ? makeMatBias(parsed.policy.passBias) : void 0;
    this.passActivation = parsed.policy.passActivation;
    this.passMul2 = parsed.policy.passMul2 ? makeMatMul(parsed.policy.passMul2) : void 0;
    this.v1 = makeConv(parsed.value.v1);
    this.v1BN = makeBn(parsed.value.v1BN);
    this.v1Activation = parsed.value.v1Activation;
    this.v2 = makeMatMul(parsed.value.v2);
    this.v2Bias = makeMatBias(parsed.value.v2Bias);
    this.v2Activation = parsed.value.v2Activation;
    this.v3 = makeMatMul(parsed.value.v3);
    this.v3Bias = makeMatBias(parsed.value.v3Bias);
    this.sv3 = makeMatMul(parsed.value.sv3);
    this.sv3Bias = makeMatBias(parsed.value.sv3Bias);
    this.ownership = makeConv(parsed.value.ownership);
  }
  forward(spatial2, global2) {
    return tf.tidy(() => {
      const trunk = this.forwardTrunk(spatial2, global2);
      let p1Out = conv2d2(trunk, this.p1);
      const g1Out = conv2d2(trunk, this.g1);
      const g1Out2 = bnAct(g1Out, this.g1BN, this.g1Activation);
      const g1Concat = poolRowsGPool(g1Out2);
      const g1Bias = tf.matMul(g1Concat, this.gpoolToBias.w);
      p1Out = p1Out.add(g1Bias.reshape([g1Bias.shape[0], 1, 1, g1Bias.shape[1]]));
      const p1Out2 = bnAct(p1Out, this.p1BN, this.p1Activation);
      const policy = conv2d2(p1Out2, this.p2);
      const policyPass = this.forwardPolicyPass(g1Concat);
      const v1Out = conv2d2(trunk, this.v1);
      const v1Out2 = bnAct(v1Out, this.v1BN, this.v1Activation);
      const v1Mean = poolRowsValueHead(v1Out2);
      let v2Out = tf.matMul(v1Mean, this.v2.w);
      v2Out = v2Out.add(this.v2Bias.b);
      v2Out = applyActivation2D(v2Out, this.v2Activation);
      let value = tf.matMul(v2Out, this.v3.w);
      value = value.add(this.v3Bias.b);
      let scoreValue = tf.matMul(v2Out, this.sv3.w);
      scoreValue = scoreValue.add(this.sv3Bias.b);
      if (this.scoreValueChannels > 4) {
        scoreValue = scoreValue.slice([0, 0], [scoreValue.shape[0], 4]);
      }
      const ownership = conv2d2(v1Out2, this.ownership);
      return { policy, policyPass, value, scoreValue, ownership };
    });
  }
  forwardPolicyValue(spatial2, global2) {
    return tf.tidy(() => {
      const trunk = this.forwardTrunk(spatial2, global2);
      let p1Out = conv2d2(trunk, this.p1);
      const g1Out = conv2d2(trunk, this.g1);
      const g1Out2 = bnAct(g1Out, this.g1BN, this.g1Activation);
      const g1Concat = poolRowsGPool(g1Out2);
      const g1Bias = tf.matMul(g1Concat, this.gpoolToBias.w);
      p1Out = p1Out.add(g1Bias.reshape([g1Bias.shape[0], 1, 1, g1Bias.shape[1]]));
      const p1Out2 = bnAct(p1Out, this.p1BN, this.p1Activation);
      const policy = conv2d2(p1Out2, this.p2);
      const policyPass = this.forwardPolicyPass(g1Concat);
      const v1Out = conv2d2(trunk, this.v1);
      const v1Out2 = bnAct(v1Out, this.v1BN, this.v1Activation);
      const v1Mean = poolRowsValueHead(v1Out2);
      let v2Out = tf.matMul(v1Mean, this.v2.w);
      v2Out = v2Out.add(this.v2Bias.b);
      v2Out = applyActivation2D(v2Out, this.v2Activation);
      let value = tf.matMul(v2Out, this.v3.w);
      value = value.add(this.v3Bias.b);
      let scoreValue = tf.matMul(v2Out, this.sv3.w);
      scoreValue = scoreValue.add(this.sv3Bias.b);
      if (this.scoreValueChannels > 4) {
        scoreValue = scoreValue.slice([0, 0], [scoreValue.shape[0], 4]);
      }
      return { policy, policyPass, value, scoreValue };
    });
  }
  forwardValueOnly(spatial2, global2) {
    return tf.tidy(() => {
      const trunk = this.forwardTrunk(spatial2, global2);
      const v1Out = conv2d2(trunk, this.v1);
      const v1Out2 = bnAct(v1Out, this.v1BN, this.v1Activation);
      const v1Mean = poolRowsValueHead(v1Out2);
      let v2Out = tf.matMul(v1Mean, this.v2.w);
      v2Out = v2Out.add(this.v2Bias.b);
      v2Out = applyActivation2D(v2Out, this.v2Activation);
      let value = tf.matMul(v2Out, this.v3.w);
      value = value.add(this.v3Bias.b);
      let scoreValue = tf.matMul(v2Out, this.sv3.w);
      scoreValue = scoreValue.add(this.sv3Bias.b);
      if (this.scoreValueChannels > 4) {
        scoreValue = scoreValue.slice([0, 0], [scoreValue.shape[0], 4]);
      }
      return { value, scoreValue };
    });
  }
  forwardTrunk(spatial2, global2) {
    let trunk = conv2d2(spatial2, this.trunkConv1);
    const ginput = tf.matMul(global2, this.trunkGInput.w);
    trunk = trunk.add(ginput.reshape([ginput.shape[0], 1, 1, ginput.shape[1]]));
    trunk = this.applyBlockStack(trunk, this.trunkBlocks);
    return bnAct(trunk, this.trunkTipBN, this.trunkTipActivation);
  }
  forwardPolicyPass(gpool) {
    let pass = tf.matMul(gpool, this.passMul.w);
    if (this.passBias && this.passActivation && this.passMul2) {
      pass = pass.add(this.passBias.b);
      pass = applyActivation2D(pass, this.passActivation);
      pass = tf.matMul(pass, this.passMul2.w);
    }
    return pass;
  }
  applyBlockStack(trunk, blocks) {
    for (const block of blocks) {
      if (block.kind === "ordinary") {
        const a2 = bnAct(trunk, block.preBN, block.preActivation);
        const b = conv2d2(a2, block.w1);
        const c2 = bnAct(b, block.midBN, block.midActivation);
        const d2 = conv2d2(c2, block.w2);
        trunk = trunk.add(d2);
        continue;
      }
      if (block.kind === "gpool") {
        const a2 = bnAct(trunk, block.preBN, block.preActivation);
        let regularOut = conv2d2(a2, block.w1a);
        const gpoolOut = conv2d2(a2, block.w1b);
        const gpoolOut2 = bnAct(gpoolOut, block.gpoolBN, block.gpoolActivation);
        const gpoolConcat = poolRowsGPool(gpoolOut2);
        const gpoolBias = tf.matMul(gpoolConcat, block.w1r.w);
        regularOut = regularOut.add(gpoolBias.reshape([gpoolBias.shape[0], 1, 1, gpoolBias.shape[1]]));
        const c2 = bnAct(regularOut, block.midBN, block.midActivation);
        const d2 = conv2d2(c2, block.w2);
        trunk = trunk.add(d2);
        continue;
      }
      const a = bnAct(trunk, block.preBN, block.preActivation);
      let mid = conv2d2(a, block.preConv);
      mid = this.applyBlockStack(mid, block.blocks);
      const c = bnAct(mid, block.postBN, block.postActivation);
      const d = conv2d2(c, block.postConv);
      trunk = trunk.add(d);
    }
    return trunk;
  }
  dispose() {
    const tensors = [
      this.trunkConv1.filter,
      this.trunkGInput.w,
      this.trunkTipBN.scale,
      this.trunkTipBN.bias,
      this.p1.filter,
      this.g1.filter,
      this.g1BN.scale,
      this.g1BN.bias,
      this.gpoolToBias.w,
      this.p1BN.scale,
      this.p1BN.bias,
      this.p2.filter,
      this.passMul.w,
      ...this.passBias ? [this.passBias.b] : [],
      ...this.passMul2 ? [this.passMul2.w] : [],
      this.v1.filter,
      this.v1BN.scale,
      this.v1BN.bias,
      this.v2.w,
      this.v2Bias.b,
      this.v3.w,
      this.v3Bias.b,
      this.sv3.w,
      this.sv3Bias.b,
      this.ownership.filter
    ];
    const pushBlockTensors = (block) => {
      tensors.push(block.preBN.scale, block.preBN.bias);
      if (block.kind === "ordinary") {
        tensors.push(block.w1.filter, block.midBN.scale, block.midBN.bias, block.w2.filter);
        return;
      }
      if (block.kind === "gpool") {
        tensors.push(
          block.w1a.filter,
          block.w1b.filter,
          block.gpoolBN.scale,
          block.gpoolBN.bias,
          block.w1r.w,
          block.midBN.scale,
          block.midBN.bias,
          block.w2.filter
        );
        return;
      }
      tensors.push(block.preConv.filter);
      for (const inner of block.blocks) pushBlockTensors(inner);
      tensors.push(block.postBN.scale, block.postBN.bias, block.postConv.filter);
    };
    for (const block of this.trunkBlocks) pushBlockTensors(block);
    tf.dispose(tensors);
  }
};

// katago-engine/fastBoard.ts
var BOARD_SIZE = 19;
var BOARD_AREA = BOARD_SIZE * BOARD_SIZE;
var PASS_MOVE = BOARD_AREA;
var NEIGHBOR_START = new Int16Array(BOARD_AREA);
var NEIGHBOR_COUNT = new Int8Array(BOARD_AREA);
var NEIGHBORS = new Int16Array(BOARD_AREA * 4);
var NEIGHBOR_STARTS = NEIGHBOR_START;
var NEIGHBOR_COUNTS = NEIGHBOR_COUNT;
var NEIGHBOR_LIST = NEIGHBORS;
var VISITED = new Int32Array(BOARD_AREA);
var LIB_VISITED = new Int32Array(BOARD_AREA);
var bfsStamp = 0;
var GROUP_BUF = new Int16Array(BOARD_AREA);
var STACK_BUF = new Int16Array(BOARD_AREA);
var PROCESSED_GROUP = new Int32Array(BOARD_AREA);
var processedStamp = 0;
var GROUP_SEEN = new Int32Array(BOARD_AREA);
var groupSeenStamp = 0;
var REGION_IDX_BY_POS = new Int16Array(BOARD_AREA);
var NEXT_EMPTY_OR_OPP = new Int16Array(BOARD_AREA);
var BORDERS_NONPASSALIVE_BY_HEADPOS = new Uint8Array(BOARD_AREA);
var GROUP_INDEX_BY_POS = new Int16Array(BOARD_AREA);
var GROUP_COLOR_BY_GROUP = new Uint8Array(BOARD_AREA);
var GROUP_START_BY_GROUP = new Int16Array(BOARD_AREA);
var GROUP_LEN_BY_GROUP = new Int16Array(BOARD_AREA);
var GROUP_STONES_FLAT = new Int16Array(BOARD_AREA);
var MAX_REGIONS = (BOARD_AREA + 1) / 2 + 1 | 0;
var REGION_HEADS = new Int16Array(MAX_REGIONS);
var VITAL_START = new Uint16Array(MAX_REGIONS);
var VITAL_LEN = new Uint8Array(MAX_REGIONS);
var NUM_INTERNAL_SPACES_MAX2 = new Uint8Array(MAX_REGIONS);
var CONTAINS_OPP = new Uint8Array(MAX_REGIONS);
var VITAL_LIST = new Int16Array(MAX_REGIONS * 4);
var REGION_QUEUE = new Int16Array(BOARD_AREA);
var PLA_GROUPS = new Int16Array(BOARD_AREA);
var PLA_GROUP_KILLED = new Uint8Array(BOARD_AREA);
var VITAL_COUNT_BY_GROUP = new Int16Array(BOARD_AREA);
var initBoardArrays = (size) => {
  BOARD_SIZE = size;
  BOARD_AREA = BOARD_SIZE * BOARD_SIZE;
  PASS_MOVE = BOARD_AREA;
  NEIGHBOR_START = new Int16Array(BOARD_AREA);
  NEIGHBOR_COUNT = new Int8Array(BOARD_AREA);
  NEIGHBORS = new Int16Array(BOARD_AREA * 4);
  let neighOffset = 0;
  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      const pos = y * BOARD_SIZE + x;
      NEIGHBOR_START[pos] = neighOffset;
      let count = 0;
      if (x > 0) {
        NEIGHBORS[neighOffset++] = pos - 1;
        count++;
      }
      if (x + 1 < BOARD_SIZE) {
        NEIGHBORS[neighOffset++] = pos + 1;
        count++;
      }
      if (y > 0) {
        NEIGHBORS[neighOffset++] = pos - BOARD_SIZE;
        count++;
      }
      if (y + 1 < BOARD_SIZE) {
        NEIGHBORS[neighOffset++] = pos + BOARD_SIZE;
        count++;
      }
      NEIGHBOR_COUNT[pos] = count;
    }
  }
  NEIGHBOR_STARTS = NEIGHBOR_START;
  NEIGHBOR_COUNTS = NEIGHBOR_COUNT;
  NEIGHBOR_LIST = NEIGHBORS;
  VISITED = new Int32Array(BOARD_AREA);
  LIB_VISITED = new Int32Array(BOARD_AREA);
  GROUP_BUF = new Int16Array(BOARD_AREA);
  STACK_BUF = new Int16Array(BOARD_AREA);
  PROCESSED_GROUP = new Int32Array(BOARD_AREA);
  GROUP_SEEN = new Int32Array(BOARD_AREA);
  bfsStamp = 0;
  processedStamp = 0;
  groupSeenStamp = 0;
  REGION_IDX_BY_POS = new Int16Array(BOARD_AREA);
  NEXT_EMPTY_OR_OPP = new Int16Array(BOARD_AREA);
  BORDERS_NONPASSALIVE_BY_HEADPOS = new Uint8Array(BOARD_AREA);
  GROUP_INDEX_BY_POS = new Int16Array(BOARD_AREA);
  GROUP_COLOR_BY_GROUP = new Uint8Array(BOARD_AREA);
  GROUP_START_BY_GROUP = new Int16Array(BOARD_AREA);
  GROUP_LEN_BY_GROUP = new Int16Array(BOARD_AREA);
  GROUP_STONES_FLAT = new Int16Array(BOARD_AREA);
  MAX_REGIONS = (BOARD_AREA + 1) / 2 + 1 | 0;
  REGION_HEADS = new Int16Array(MAX_REGIONS);
  VITAL_START = new Uint16Array(MAX_REGIONS);
  VITAL_LEN = new Uint8Array(MAX_REGIONS);
  NUM_INTERNAL_SPACES_MAX2 = new Uint8Array(MAX_REGIONS);
  CONTAINS_OPP = new Uint8Array(MAX_REGIONS);
  VITAL_LIST = new Int16Array(MAX_REGIONS * 4);
  REGION_QUEUE = new Int16Array(BOARD_AREA);
  PLA_GROUPS = new Int16Array(BOARD_AREA);
  PLA_GROUP_KILLED = new Uint8Array(BOARD_AREA);
  VITAL_COUNT_BY_GROUP = new Int16Array(BOARD_AREA);
  LADDER_STACK_SIZE = BOARD_AREA * 3 / 2 + 2 | 0;
  LADDER_SCRATCH = {
    bufMoves: new Int16Array(LADDER_BUF_SIZE),
    moveListStarts: new Int32Array(LADDER_STACK_SIZE),
    moveListLens: new Int32Array(LADDER_STACK_SIZE),
    moveListCur: new Int32Array(LADDER_STACK_SIZE),
    recordMoves: new Int16Array(LADDER_STACK_SIZE),
    recordPlayers: new Uint8Array(LADDER_STACK_SIZE),
    recordKoPointBefore: new Int16Array(LADDER_STACK_SIZE),
    recordCaptureStart: new Int32Array(LADDER_STACK_SIZE),
    tmpKoPointBefore: new Int16Array(1),
    tmpCaptureStart: new Int32Array(1),
    captureStack: []
  };
  LADDER_GROUP_SEEN = new Int32Array(BOARD_AREA);
  LADDER_OPP_GROUP_SEEN = new Int32Array(BOARD_AREA);
  LADDER_GROUP_COPY = new Int16Array(BOARD_AREA);
  LADDER_CONNECT_GROUP_SEEN = new Int32Array(BOARD_AREA);
  LADDER_CAPTURED = new Int32Array(BOARD_AREA);
  ladderGroupSeenStamp = 0;
  ladderOppGroupSeenStamp = 0;
  ladderConnectGroupSeenStamp = 0;
  ladderCapturedStamp = 0;
  LADDER_FEATURES_SCRATCH_V7 = {
    copyPos: { stones: new Uint8Array(BOARD_AREA), koPoint: -1 },
    groupStones: new Int16Array(BOARD_AREA),
    workingMoves: []
  };
};
var LADDER_STACK_SIZE = BOARD_AREA * 3 / 2 + 2 | 0;
var LADDER_BUF_SIZE = 8192;
var LADDER_SCRATCH = {
  bufMoves: new Int16Array(LADDER_BUF_SIZE),
  moveListStarts: new Int32Array(LADDER_STACK_SIZE),
  moveListLens: new Int32Array(LADDER_STACK_SIZE),
  moveListCur: new Int32Array(LADDER_STACK_SIZE),
  recordMoves: new Int16Array(LADDER_STACK_SIZE),
  recordPlayers: new Uint8Array(LADDER_STACK_SIZE),
  recordKoPointBefore: new Int16Array(LADDER_STACK_SIZE),
  recordCaptureStart: new Int32Array(LADDER_STACK_SIZE),
  tmpKoPointBefore: new Int16Array(1),
  tmpCaptureStart: new Int32Array(1),
  captureStack: []
};
var LADDER_GROUP_SEEN = new Int32Array(BOARD_AREA);
var ladderGroupSeenStamp = 0;
var LADDER_OPP_GROUP_SEEN = new Int32Array(BOARD_AREA);
var ladderOppGroupSeenStamp = 0;
var LADDER_GROUP_COPY = new Int16Array(BOARD_AREA);
var LADDER_CONNECT_GROUP_SEEN = new Int32Array(BOARD_AREA);
var ladderConnectGroupSeenStamp = 0;
var LADDER_CAPTURED = new Int32Array(BOARD_AREA);
var ladderCapturedStamp = 0;
var LADDER_FEATURES_SCRATCH_V7 = {
  copyPos: { stones: new Uint8Array(BOARD_AREA), koPoint: -1 },
  groupStones: new Int16Array(BOARD_AREA),
  workingMoves: []
};
initBoardArrays(BOARD_SIZE);

// katago-engine/src_gameLogic.ts
var getOpponent = (player) => player === "black" ? "white" : "black";

// katago-engine/featuresV7.ts
var INPUT_SPATIAL_CHANNELS_V7 = 22;
function createKataGoInputsV7Scratch() {
  const n = BOARD_SIZE * BOARD_SIZE;
  return {
    stones: new Uint8Array(n),
    visited: new Uint8Array(n),
    libertyMarked: new Uint8Array(n),
    stack: [],
    group: [],
    touchedLibs: []
  };
}
var idxNHWC = (x, y, c) => (y * BOARD_SIZE + x) * INPUT_SPATIAL_CHANNELS_V7 + c;
function fillInputsV7(args) {
  const { board, currentPlayer, moveHistory, komi } = args;
  const rules = args.rules ?? "japanese";
  const pla = currentPlayer;
  const opp = getOpponent(pla);
  const spatial2 = args.outSpatial;
  const global2 = args.outGlobal;
  spatial2.fill(0);
  global2.fill(0);
  for (let pos = 0; pos < BOARD_SIZE * BOARD_SIZE; pos++) spatial2[pos * INPUT_SPATIAL_CHANNELS_V7 + 0] = 1;
  const scratch2 = args.scratch ?? createKataGoInputsV7Scratch();
  const stones = scratch2.stones;
  stones.fill(0);
  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      const v = board[y][x];
      if (v === null) continue;
      stones[y * BOARD_SIZE + x] = v === "black" ? 1 : 2;
      if (v === pla) spatial2[idxNHWC(x, y, 1)] = 1;
      else spatial2[idxNHWC(x, y, 2)] = 1;
    }
  }
  const visited = scratch2.visited;
  const libertyMarked = scratch2.libertyMarked;
  const stack = scratch2.stack;
  const group = scratch2.group;
  const touchedLibs = scratch2.touchedLibs;
  visited.fill(0);
  libertyMarked.fill(0);
  for (let pos = 0; pos < stones.length; pos++) {
    const color = stones[pos];
    if (color === 0) continue;
    if (visited[pos]) continue;
    visited[pos] = 1;
    stack.length = 0;
    group.length = 0;
    touchedLibs.length = 0;
    stack.push(pos);
    group.push(pos);
    let liberties = 0;
    while (stack.length > 0) {
      const p = stack.pop();
      const x = p % BOARD_SIZE;
      const y = p / BOARD_SIZE | 0;
      if (x + 1 < BOARD_SIZE) {
        const npos = p + 1;
        const ncolor = stones[npos];
        if (ncolor === 0) {
          if (!libertyMarked[npos]) {
            libertyMarked[npos] = 1;
            touchedLibs.push(npos);
            liberties++;
          }
        } else if (ncolor === color && !visited[npos]) {
          visited[npos] = 1;
          stack.push(npos);
          group.push(npos);
        }
      }
      if (x > 0) {
        const npos = p - 1;
        const ncolor = stones[npos];
        if (ncolor === 0) {
          if (!libertyMarked[npos]) {
            libertyMarked[npos] = 1;
            touchedLibs.push(npos);
            liberties++;
          }
        } else if (ncolor === color && !visited[npos]) {
          visited[npos] = 1;
          stack.push(npos);
          group.push(npos);
        }
      }
      if (y + 1 < BOARD_SIZE) {
        const npos = p + BOARD_SIZE;
        const ncolor = stones[npos];
        if (ncolor === 0) {
          if (!libertyMarked[npos]) {
            libertyMarked[npos] = 1;
            touchedLibs.push(npos);
            liberties++;
          }
        } else if (ncolor === color && !visited[npos]) {
          visited[npos] = 1;
          stack.push(npos);
          group.push(npos);
        }
      }
      if (y > 0) {
        const npos = p - BOARD_SIZE;
        const ncolor = stones[npos];
        if (ncolor === 0) {
          if (!libertyMarked[npos]) {
            libertyMarked[npos] = 1;
            touchedLibs.push(npos);
            liberties++;
          }
        } else if (ncolor === color && !visited[npos]) {
          visited[npos] = 1;
          stack.push(npos);
          group.push(npos);
        }
      }
    }
    for (const npos of touchedLibs) libertyMarked[npos] = 0;
    const plane = liberties === 1 ? 3 : liberties === 2 ? 4 : liberties === 3 ? 5 : -1;
    if (plane >= 0) {
      for (const gpos of group) {
        const gx = gpos % BOARD_SIZE;
        const gy = gpos / BOARD_SIZE | 0;
        spatial2[idxNHWC(gx, gy, plane)] = 1;
      }
    }
  }
  const lastMove = moveHistory.length > 0 ? moveHistory[moveHistory.length - 1] : null;
  const passWouldEndGame = !!lastMove && (lastMove.x === -1 || lastMove.y === -1);
  const suppressHistory = args.conservativePassAndIsRoot === true && passWouldEndGame;
  const historyPlanes = [9, 10, 11, 12, 13];
  const passGlobals = [0, 1, 2, 3, 4];
  const expectedPlayers = [opp, pla, opp, pla, opp];
  if (!suppressHistory) {
    for (let i = 0; i < 5; i++) {
      const m = moveHistory[moveHistory.length - 1 - i];
      if (!m) break;
      if (m.player !== expectedPlayers[i]) break;
      if (m.x === -1 || m.y === -1) {
        global2[passGlobals[i]] = 1;
      } else {
        spatial2[idxNHWC(m.x, m.y, historyPlanes[i])] = 1;
      }
    }
  }
  const selfKomi = pla === "white" ? komi : -komi;
  global2[5] = selfKomi / 20;
  if (rules === "japanese" || rules === "korean") {
    global2[9] = 1;
    global2[10] = 1;
  }
  global2[14] = !suppressHistory && passWouldEndGame ? 1 : 0;
  if (rules === "chinese") {
    const boardAreaIsEven = BOARD_SIZE * BOARD_SIZE % 2 === 0;
    const drawableKomisAreEven = boardAreaIsEven;
    let komiFloor;
    if (drawableKomisAreEven) komiFloor = Math.floor(selfKomi / 2) * 2;
    else komiFloor = Math.floor((selfKomi - 1) / 2) * 2 + 1;
    let delta = selfKomi - komiFloor;
    if (delta < 0) delta = 0;
    if (delta > 2) delta = 2;
    let wave;
    if (delta < 0.5) wave = delta;
    else if (delta < 1.5) wave = 1 - delta;
    else wave = delta - 2;
    global2[18] = wave;
  }
}

// katago-engine/evalV8.ts
var softPlus = (x) => {
  if (x > 20) return x;
  if (x < -20) return Math.exp(x);
  return Math.log1p(Math.exp(x));
};
function postprocessKataGoV8(args) {
  const { nextPlayer, valueLogits, scoreValue } = args;
  const postProcessParams = args.postProcessParams;
  const outputScaleMultiplier = postProcessParams?.outputScaleMultiplier ?? 1;
  const winLogits = valueLogits[0] * outputScaleMultiplier;
  const lossLogits = valueLogits[1] * outputScaleMultiplier;
  const noResultLogits = valueLogits[2] * outputScaleMultiplier;
  const maxLogits = Math.max(winLogits, lossLogits, noResultLogits);
  let winProb = Math.exp(winLogits - maxLogits);
  let lossProb = Math.exp(lossLogits - maxLogits);
  let noResultProb = Math.exp(noResultLogits - maxLogits);
  const probSum = winProb + lossProb + noResultProb;
  winProb /= probSum;
  lossProb /= probSum;
  noResultProb /= probSum;
  const scoreMeanMultiplier = postProcessParams?.scoreMeanMultiplier ?? 20;
  const scoreStdevMultiplier = postProcessParams?.scoreStdevMultiplier ?? 20;
  const leadMultiplier = postProcessParams?.leadMultiplier ?? 20;
  const scoreMeanPreScaled = scoreValue[0] * outputScaleMultiplier;
  const scoreStdevPreSoftplus = scoreValue[1] * outputScaleMultiplier;
  const leadPreScaled = scoreValue[2] * outputScaleMultiplier;
  let scoreMean = scoreMeanPreScaled * scoreMeanMultiplier;
  const scoreStdev = softPlus(scoreStdevPreSoftplus) * scoreStdevMultiplier;
  let scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;
  let lead = leadPreScaled * leadMultiplier;
  scoreMean *= 1 - noResultProb;
  scoreMeanSq *= 1 - noResultProb;
  lead *= 1 - noResultProb;
  const blackWinProb = nextPlayer === "black" ? winProb : lossProb;
  const blackScoreLead = nextPlayer === "black" ? lead : -lead;
  const blackScoreMean = nextPlayer === "black" ? scoreMean : -scoreMean;
  const blackScoreStdev = Math.sqrt(Math.max(0, scoreMeanSq - scoreMean * scoreMean));
  const blackNoResultProb = noResultProb;
  return {
    blackWinProb,
    blackScoreLead,
    blackScoreMean,
    blackScoreStdev,
    blackNoResultProb
  };
}

// katago-engine/katago-entry.ts
var model = null;
var modelName = "";
async function ungzip(data) {
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([data]).stream().pipeThrough(ds);
  const ab = await new Response(stream).arrayBuffer();
  return new Uint8Array(ab);
}
async function initKataGo(modelBytes) {
  const isGzip = modelBytes[0] === 31 && modelBytes[1] === 139;
  const data = isGzip ? await ungzip(modelBytes) : modelBytes;
  const parsed = parseKataGoModelV8(data);
  model = new KataGoModelV8Tf(parsed);
  modelName = parsed.modelName;
  const ws = tf2.zeros([1, BOARD_SIZE, BOARD_SIZE, 22], "float32");
  const wg = tf2.zeros([1, BOARD_SIZE], "float32");
  const out = model.forwardValueOnly(ws, wg);
  await Promise.all([out.value.data(), out.scoreValue.data()]);
  ws.dispose();
  wg.dispose();
  out.value.dispose();
  out.scoreValue.dispose();
}
function modelReady() {
  return model !== null;
}
function getModelName() {
  return modelName;
}
function buildInputs(boardMap, moves, currentColor) {
  const board = [];
  for (let y = 0; y < BOARD_SIZE; y++) {
    const row = [];
    for (let x = 0; x < BOARD_SIZE; x++) row.push(null);
    board.push(row);
  }
  boardMap.forEach((s, k) => {
    const p = k.split(",");
    board[+p[1]][+p[0]] = s.color === "B" ? "black" : "white";
  });
  const currentPlayer = currentColor === "B" ? "black" : "white";
  const moveHistory = moves.map((m) => ({
    x: m.x,
    y: m.y,
    player: m.color === "B" ? "black" : "white"
  }));
  return { board, currentPlayer, moveHistory };
}
var scratch = createKataGoInputsV7Scratch();
var spatial = new Float32Array(BOARD_SIZE * BOARD_SIZE * 22);
var global = new Float32Array(BOARD_SIZE);
async function evaluatePosition(boardMap, moves, currentColor, komi) {
  if (!model) throw new Error("KataGo \u672A\u521D\u59CB\u5316");
  const { board, currentPlayer, moveHistory } = buildInputs(boardMap, moves, currentColor);
  spatial.fill(0);
  global.fill(0);
  fillInputsV7({ board, currentPlayer, moveHistory, komi, outSpatial: spatial, outGlobal: global, scratch });
  const s = tf2.tensor4d(spatial, [1, BOARD_SIZE, BOARD_SIZE, 22]);
  const g = tf2.tensor2d(global, [1, BOARD_SIZE]);
  const out = model.forward(s, g);
  const [polData, passData, valData, scoreData] = await Promise.all([
    out.policy.data(),
    out.policyPass.data(),
    out.value.data(),
    out.scoreValue.data()
  ]);
  s.dispose();
  g.dispose();
  out.policy.dispose();
  out.policyPass.dispose();
  out.value.dispose();
  out.scoreValue.dispose();
  out.ownership.dispose();
  const C = model.policyOutChannels;
  const policy = new Float32Array(BOARD_AREA);
  for (let p = 0; p < BOARD_AREA; p++) policy[p] = polData[p * C];
  const passLogit = passData[0];
  const ev = postprocessKataGoV8({
    nextPlayer: currentPlayer,
    valueLogits: Array.from(valData),
    scoreValue: Array.from(scoreData),
    postProcessParams: model.postProcessParams
  });
  const curWinRate = currentPlayer === "black" ? ev.blackWinProb : 1 - ev.blackWinProb;
  const curScoreLead = currentPlayer === "black" ? ev.blackScoreLead : -ev.blackScoreLead;
  return { policy, passLogit, winRate: curWinRate, scoreLead: curScoreLead };
}
async function pickMove(boardMap, moves, currentColor, legalMoves, komi, valuePly = true) {
  if (!legalMoves.length) return null;
  const ev = await evaluatePosition(boardMap, moves, currentColor, komi);
  let maxLogit = -Infinity;
  for (const lm of legalMoves) {
    const l = ev.policy[lm.y * BOARD_SIZE + lm.x];
    if (l > maxLogit) maxLogit = l;
  }
  const cand = legalMoves.slice().sort((a, b) => {
    return ev.policy[b.y * BOARD_SIZE + b.x] - ev.policy[a.y * BOARD_SIZE + a.x];
  }).slice(0, 8);
  if (!valuePly || cand.length === 1) return cand[0];
  return cand[0];
}
async function pickMove1Ply(boardMap, moves, currentColor, legalMoves, komi, simulate) {
  if (!legalMoves.length) return null;
  const ev = await evaluatePosition(boardMap, moves, currentColor, komi);
  const cand = legalMoves.slice().sort(
    (a, b) => ev.policy[b.y * BOARD_SIZE + b.x] - ev.policy[a.y * BOARD_SIZE + a.x]
  ).slice(0, 6);
  let best = cand[0], bestVal = -Infinity;
  const opp = currentColor === "B" ? "W" : "B";
  for (const c of cand) {
    const { boardMap: nm, moves: nmv } = simulate(c.x, c.y);
    try {
      const e2 = await evaluatePosition(nm, nmv, opp, komi);
      const val = e2.winRate;
      const score = -e2.scoreLead;
      const blended = -val * 1;
      if (blended > bestVal) {
        bestVal = blended;
        best = c;
      }
    } catch (e) {
    }
  }
  return best;
}
export {
  evaluatePosition,
  getModelName,
  initKataGo,
  modelReady,
  pickMove,
  pickMove1Ply
};
