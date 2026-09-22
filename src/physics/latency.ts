// ============================================================================
//  latency.ts — Physically-derived performance model
//
//  Nothing here is a made-up number: every quantity is computed from an
//  explicit closed-form expression using published silicon-photonics device
//  parameters, and the formula is shown next to the result in the UI.
//
//  The honest boundary of the claim: the OPTICAL part is time-of-flight
//  limited (tens of ps). The end-to-end latency of a real accelerator is
//  dominated by the electro-optic and opto-electronic conversion, which is
//  why the sub-microsecond figure holds but the sub-nanosecond one does not.
// ============================================================================

export const C_LIGHT = 299_792_458; // m/s

export interface HardwareParams {
  /** group index of a Si strip waveguide at 1550 nm */
  ng: number;
  /** physical length of one MZI cell, um */
  mziPitch: number;
  /** propagation loss, dB/cm */
  lossDbPerCm: number;
  /** insertion loss per directional coupler pair, dB */
  mziLossDb: number;
  /** symbol / modulation rate, GHz */
  modulationGHz: number;
  /** DAC + driver latency, ps */
  eoLatencyPs: number;
  /** photodiode + TIA + ADC latency, ps */
  oeLatencyPs: number;
  /** static power per phase shifter, mW (thermo-optic vs MEMS) */
  shifterMw: number;
  /** laser + comb source power, mW */
  sourceMw: number;
  /** number of WDM channels multiplexed through the same mesh */
  wdmChannels: number;
}

export const HW_PRESETS: Record<string, { label: string; note: string; p: HardwareParams }> = {
  thermo: {
    label: 'Thermo-optique (mature)',
    note: 'Déphaseurs chauffants — robustes, mais ~10 mW chacun en statique.',
    p: { ng: 4.2, mziPitch: 250, lossDbPerCm: 2.0, mziLossDb: 0.12, modulationGHz: 25, eoLatencyPs: 250, oeLatencyPs: 300, shifterMw: 10, sourceMw: 120, wdmChannels: 8 },
  },
  mems: {
    label: 'MEMS / plasma-dispersion',
    note: 'Déphaseurs quasi non-dissipatifs, mais surface et tension plus élevées.',
    p: { ng: 4.2, mziPitch: 180, lossDbPerCm: 1.5, mziLossDb: 0.08, modulationGHz: 50, eoLatencyPs: 150, oeLatencyPs: 200, shifterMw: 0.02, sourceMw: 120, wdmChannels: 16 },
  },
  sin: {
    label: 'Nitrure de silicium (bas pertes)',
    note: 'Pertes 0.1 dB/cm → maillages beaucoup plus grands, mais indice de groupe plus faible.',
    p: { ng: 1.9, mziPitch: 600, lossDbPerCm: 0.1, mziLossDb: 0.05, modulationGHz: 20, eoLatencyPs: 250, oeLatencyPs: 300, shifterMw: 8, sourceMw: 150, wdmChannels: 32 },
  },
};

export interface MeshPerformance {
  n: number;
  depth: number;
  mziCount: number;
  opticalLengthMm: number;
  timeOfFlightPs: number;
  totalLatencyPs: number;
  insertionLossDb: number;
  transmission: number;
  /** real FLOPs per unitary application: complex MAC = 8 real ops */
  flopsPerPass: number;
  peakTflops: number;
  staticPowerMw: number;
  totalPowerMw: number;
  tflopsPerWatt: number;
  formulas: { label: string; expr: string; value: string }[];
}

export function meshPerformance(n: number, depth: number, mziCount: number, p: HardwareParams): MeshPerformance {
  const lengthUm = depth * p.mziPitch;
  const lengthMm = lengthUm / 1000;
  const lengthM = lengthUm * 1e-6;

  const tofPs = ((p.ng * lengthM) / C_LIGHT) * 1e12;
  const totalPs = tofPs + p.eoLatencyPs + p.oeLatencyPs;

  const propLoss = (p.lossDbPerCm * lengthUm) / 10000;
  const mziLoss = depth * p.mziLossDb;
  const lossDb = propLoss + mziLoss;
  const transmission = Math.pow(10, -lossDb / 10);

  const flopsPerPass = 8 * n * n * p.wdmChannels;
  const peakFlops = flopsPerPass * p.modulationGHz * 1e9;
  const peakTflops = peakFlops / 1e12;

  const staticMw = mziCount * 2 * p.shifterMw; // theta + phi per cell
  const totalMw = staticMw + p.sourceMw;
  const tflopsPerWatt = peakTflops / (totalMw / 1000);

  return {
    n, depth, mziCount,
    opticalLengthMm: lengthMm,
    timeOfFlightPs: tofPs,
    totalLatencyPs: totalPs,
    insertionLossDb: lossDb,
    transmission,
    flopsPerPass,
    peakTflops,
    staticPowerMw: staticMw,
    totalPowerMw: totalMw,
    tflopsPerWatt,
    formulas: [
      { label: 'Longueur optique', expr: 'L = profondeur × pas_MZI', value: `${depth} × ${p.mziPitch} µm = ${lengthMm.toFixed(2)} mm` },
      { label: 'Temps de vol', expr: 't = n_g · L / c', value: `${p.ng} × ${lengthMm.toFixed(2)}mm / c = ${tofPs.toFixed(1)} ps` },
      { label: 'Latence bout-en-bout', expr: 't + t_EO + t_OE', value: `${tofPs.toFixed(0)} + ${p.eoLatencyPs} + ${p.oeLatencyPs} = ${totalPs.toFixed(0)} ps` },
      { label: 'Pertes d’insertion', expr: 'α·L + profondeur·IL_MZI', value: `${propLoss.toFixed(2)} + ${mziLoss.toFixed(2)} = ${lossDb.toFixed(2)} dB` },
      { label: 'Opérations / passage', expr: '8·N²·N_WDM (MAC complexe)', value: `8×${n}²×${p.wdmChannels} = ${flopsPerPass.toLocaleString('fr-FR')} FLOP` },
      { label: 'Débit crête', expr: 'FLOP/passage × f_mod', value: `× ${p.modulationGHz} GHz = ${peakTflops.toFixed(2)} TFLOP/s` },
      { label: 'Puissance statique', expr: '2 · N_MZI · P_shifter', value: `2×${mziCount}×${p.shifterMw} mW = ${staticMw.toFixed(1)} mW` },
      { label: 'Efficacité', expr: 'TFLOP/s ÷ (P_stat + P_laser)', value: `${tflopsPerWatt.toFixed(1)} TFLOP/s/W` },
    ],
  };
}

/** Number of round trips a CIM needs, and the corresponding physical wall time. */
export function cimPhysicalTime(iterations: number, cavityLengthM: number, ng = 1.468, fpgaLatencyNs = 30) {
  const roundTripNs = ((ng * cavityLengthM) / C_LIGHT) * 1e9;
  const perTripNs = roundTripNs + fpgaLatencyNs;
  return {
    roundTripNs,
    perTripNs,
    totalUs: (iterations * perTripNs) / 1000,
    formula: `T = N_tours × (n_g·L_cav/c + t_FPGA) = ${iterations} × (${roundTripNs.toFixed(1)} + ${fpgaLatencyNs}) ns`,
  };
}
