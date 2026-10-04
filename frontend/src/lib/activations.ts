// Activation registry for the Act 1 gallery. Data-driven: adding a function is a ONE-ENTRY change in ACTIVATIONS.
// Python twin: backend/app/activations.py (names match).

export interface ParamSpec {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface ActivationDef {
  name: string;
  label: string;
  formula: string;
  f: (x: number, p: Record<string, number>) => number;
  df: (x: number, p: Record<string, number>) => number;
  params?: ParamSpec;
  /** slope is zero everywhere: cannot learn (shows the red "broke" badge) */
  dead?: boolean;
  note?: string;
}

function erf(x: number): number {
  // Abramowitz-Stegun 7.1.26 (max error 1.5e-7)
  const s = Math.sign(x);
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const ACTIVATIONS: ActivationDef[] = [
  { name: "relu", label: "ReLU", formula: "max(0, x)", f: (x) => (x > 0 ? x : 0), df: (x) => (x > 0 ? 1 : 0) },
  { name: "sigmoid", label: "Sigmoid", formula: "1 / (1 + e⁻ˣ)", f: (x) => sigmoid(x), df: (x) => sigmoid(x) * (1 - sigmoid(x)) },
  { name: "tanh", label: "Tanh", formula: "tanh(x)", f: (x) => Math.tanh(x), df: (x) => 1 - Math.tanh(x) ** 2 },
  {
    name: "gelu", label: "GELU", formula: "x · Φ(x)",
    f: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)),
    df: (x) => 0.5 * (1 + erf(x / Math.SQRT2)) + (x * Math.exp((-x * x) / 2)) / Math.sqrt(2 * Math.PI),
  },
  { name: "hard_step", label: "Hard step", formula: "1 if x > 0 else 0", f: (x) => (x > 0 ? 1 : 0), df: () => 0, dead: true,
    note: "Its slope is zero everywhere, so nothing can be learned." },
  {
    name: "leaky_relu", label: "Leaky ReLU", formula: "x if x > 0 else α·x",
    f: (x, p) => (x > 0 ? x : p.alpha * x), df: (x, p) => (x > 0 ? 1 : p.alpha),
    params: { key: "alpha", label: "α", min: 0, max: 1, step: 0.01, default: 0.1 },
    note: "At α = 1 it is a straight line, and the collapse returns.",
  },
  {
    name: "swish", label: "Swish / SiLU", formula: "x · σ(β·x)",
    f: (x, p) => x * sigmoid(p.beta * x), df: (x, p) => sigmoid(p.beta * x) + p.beta * x * sigmoid(p.beta * x) * (1 - sigmoid(p.beta * x)),
    params: { key: "beta", label: "β", min: 0.1, max: 5, step: 0.1, default: 1 },
  },
  {
    name: "softplus", label: "Softplus", formula: "ln(1 + e^(β·x)) / β",
    f: (x, p) => (p.beta * x > 30 ? x : Math.log1p(Math.exp(p.beta * x)) / p.beta), df: (x, p) => sigmoid(p.beta * x),
    params: { key: "beta", label: "β", min: 0.1, max: 5, step: 0.1, default: 1 },
  },
  {
    name: "elu", label: "ELU", formula: "x if x > 0 else α·(eˣ − 1)",
    f: (x, p) => (x > 0 ? x : p.alpha * (Math.exp(x) - 1)), df: (x, p) => (x > 0 ? 1 : p.alpha * Math.exp(x)),
    params: { key: "alpha", label: "α", min: 0.1, max: 3, step: 0.1, default: 1 },
  },
  {
    name: "clipped_relu", label: "Clipped ReLU", formula: "min(cap, max(0, x))",
    f: (x, p) => Math.min(p.cap, Math.max(0, x)), df: (x, p) => (x > 0 && x < p.cap ? 1 : 0),
    params: { key: "cap", label: "cap", min: 0.5, max: 6, step: 0.1, default: 3 },
  },
];

export const LINEAR: ActivationDef = { name: "linear", label: "none (linear)", formula: "x", f: (x) => x, df: () => 1 };

export function getActivation(name: string): ActivationDef {
  if (name === "linear") return LINEAR;
  const a = ACTIVATIONS.find((x) => x.name === name);
  if (!a) throw new Error(`unknown activation ${name}`);
  return a;
}

export function defaultParams(a: ActivationDef): Record<string, number> {
  return a.params ? { [a.params.key]: a.params.default } : {};
}
