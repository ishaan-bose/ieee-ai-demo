// The site is an ordered list of stages (SPEC 5.1). Each stage declares which UI elements it UNLOCKS; an element that is not
// unlocked does not exist on screen, and pops in when its stage is reached. Going back removes later elements again.
export interface Stage {
  id: string;
  act: 0 | 1 | 2 | 3 | 4;
  title: string;
  unlocks: string[];
  /** keys hint shown in the corner for this stage */
  hint?: string;
  /** talk track for the presenter's phone (/presenter) */
  notes: string[];
}

export const STAGES: Stage[] = [
  { id: "a0-duel", act: 0, title: "The duel: you vs. a random AI", unlocks: ["doodle", "options", "score", "gauge"], hint: "Space: start · 1-4: answer",
    notes: ["Pick a volunteer. They play five doodle rounds against \"the AI\".", "The AI is a random guesser: it answers at the same moment as the human and has no idea what a cat is.", "Let them win. Point at the score: this is what chance looks like.", "Say: \"By the end of this talk you will see an AI that is not guessing.\""] },
  { id: "a1-clusters", act: 1, title: "Draw a line", unlocks: ["plot", "line", "gauge", "guess"], hint: "Space: guess again",
    notes: ["Two groups of dots. A random line splits them: the gauge sits near 50%, which is chance for two groups.", "Press Space to guess again. Each guess is just a random line.", "Ask: \"how would you find a better line?\""] },
  { id: "a1-knobs", act: 1, title: "Three knobs", unlocks: ["sliders", "equation", "findbest", "word-parameters"], hint: "Space: find best knobs",
    notes: ["A line is three numbers: tilt, tilt, slide. Those numbers are called PARAMETERS.", "Drag the sliders by hand first, then press Space: the computer finds the best three numbers with a tiny bit of algebra.", "Key word: parameters. A model is just parameters + an equation."] },
  { id: "a1-spirals", act: 1, title: "Spirals", unlocks: ["spirals"], hint: "Space: find best knobs",
    notes: ["The dots morph into two spirals. No straight line can separate these.", "The gauge goes red. The best three knobs still fail.", "\"What if we use MORE lines?\""] },
  { id: "a1-depth", act: 1, title: "Stack more lines (layers)", unlocks: ["depth", "train", "blocks"], hint: "Space: train · slider: depth",
    notes: ["Each block is a layer. Use the depth slider and train.", "Stack as many as you like: the boundary stays one straight line. Make them guess why.", "(The reveal comes next.)"] },
  { id: "a1-fuse", act: 1, title: "Why stacking lines does nothing", unlocks: ["fuse"], hint: "Space: fuse the blocks",
    notes: ["Click Space: all the stacked blocks fuse into ONE block, W₃·W₂·W₁ = W.", "A line of a line of a line is still a line. Stacking linear layers adds nothing.", "We need a bend."] },
  { id: "a1-relu", act: 1, title: "A bend: ReLU", unlocks: ["relu", "width", "relu-plot"], hint: "Space: train · sliders: depth, width",
    notes: ["ReLU is max(0, x): flat, then a slope. One small bend between layers.", "Train: the boundary bends around the spirals. This is the whole trick of deep learning.", "Try depth 1 / width 2: it still fails. Capacity matters. Then crank it up."] },
  { id: "a1-gallery", act: 1, title: "Activation gallery", unlocks: ["gallery"], hint: "click a card · Space: try it",
    notes: ["Different bends: sigmoid, tanh, GELU, and more. Each card shows the function and its slope.", "Hard step: the slope is zero everywhere, so the network cannot learn. Watch the red BROKE badge.", "Leaky ReLU at α = 1 is a straight line again: the collapse is back."] },
  { id: "a2-forward", act: 2, title: "Forward pass: it looks at your doodle", unlocks: ["draw", "network", "top5"], hint: "draw with the mouse · R: clear",
    notes: ["Volunteer draws. After every stroke the AI re-guesses: pixels flow through the network as light.", "Top-5 bars update after each stroke. This model is bundled in the page, it works offline.", "Try something ambiguous: a star vs. lightning, a sword vs. a tree."] },
  { id: "a2-loss", act: 2, title: "Loss race: how should it learn?", unlocks: ["lanes-loss", "probewall"], hint: "pick up to 3 losses · Space: race · S: recorded",
    notes: ["The audience picks up to three ways to measure a mistake (a LOSS). Same network, same data, same time on a real GPU.", "The wall shows 16 doodles the AI has never seen: red turns green when a lane gets it right.", "Loss curves are separate: different losses are not comparable numbers. Use accuracy to compare."] },
  { id: "a2-backward", act: 2, title: "Backward pass and step size", unlocks: ["backward", "valley"], hint: "slider: step size · Space: roll",
    notes: ["Learning = rolling downhill. The backward pass tells each knob which way is down.", "The valley is a CARTOON of the real landscape (millions of dimensions!). Small step: crawl. Good step: glide. Too big: overshoot.", "Next: a real race with three step sizes."] },
  { id: "a2-lr", act: 2, title: "Learning-rate race", unlocks: ["lanes-lr"], hint: "Space: race · S: recorded",
    notes: ["Three step sizes: too small, good, and the audience's pick.", "Too small barely moves; too large explodes or bounces.", "This one number is the most important knob."] },
  { id: "a2-batch", act: 2, title: "Batch-size race", unlocks: ["lanes-batch"], hint: "Space: race · S: recorded",
    notes: ["Batch 1: update after every single doodle. Full batch: one careful update after seeing ALL the doodles.", "Same learning rate, same wall-clock time. Look at the counters: updates done and doodles seen.", "Neither extreme wins: the middle is a trade-off."] },
  { id: "a3-preprocess", act: 3, title: "Hearing: raw sound vs. a spectrogram", unlocks: ["mic", "spectrogram", "models"], hint: "hold T or the button: speak · , . clips · Space: next step",
    notes: ["Hold the button and say one word (yes, no, up, down, left, right, on, off, stop, go). If the mic fails, use the sample clips.", "Two AIs hear the SAME recording: one gets the raw waveform and flails; one gets a spectrogram and works.", "Step view: waveform → windows → Fourier transform → mel scale → log. Preparing the data is half the job."] },
  { id: "a3-trap", act: 3, title: "The metrics trap", unlocks: ["trap"], hint: "Space: next reveal",
    notes: ["A detector for the word \"marvin\", which is only 2% of the data.", "\"Always say no\" scores about 98% accuracy and finds nothing. Accuracy alone can lie.", "Reveal the confusion matrix and recall: that is the honest number."] },
  { id: "a3-overfit", act: 3, title: "Overfitting", unlocks: ["overfit"], hint: "slider: training-set size · Space: replay",
    notes: ["A big model on a tiny set memorises it: training accuracy goes up, validation (new data) does not.", "Slide the training-set size: more data closes the gap.", "This is why research is about data, not just models."] },
  { id: "a4-tech", act: 4, title: "The full control panel", unlocks: ["techtree"], hint: "Space: unlock the next knob",
    notes: ["Everything you saw is a knob. There are more: optimiser, initialisation, normalisation, schedules.", "At our booth YOU get the full control panel to design a chess-evaluating network.", "Tournament after the event: your bot plays everyone else's."] },
  { id: "a4-rematch", act: 4, title: "Rematch: you vs. the trained AI", unlocks: ["doodle", "options", "score", "gauge", "rematch"], hint: "Space: start · 1-4: answer",
    notes: ["Same duel as the start. This time the AI is the trained network and it sees only the strokes drawn so far.", "Compare with your first score.", "It is not magic: it is the knobs you just learned."] },
  { id: "a4-close", act: 4, title: "That's the club", unlocks: ["stats", "booth", "pitch"], hint: "",
    notes: ["Session stats: runs, parameters trained, compute used.", "Point to the booth QR/link: build your own chess AI. Tournament after the event.", "One line: we do AI research that makes training cheaper, faster or more accurate.", "Hand back to the slides."] },
];

export const stageIndex = (id: string) => STAGES.findIndex((s) => s.id === id);

/** Elements unlocked at stage `index`: the union of `unlocks` of every stage up to and including it. */
export function unlockedAt(index: number): Set<string> {
  const s = new Set<string>();
  for (let i = 0; i <= index; i++) for (const u of STAGES[i].unlocks) s.add(u);
  return s;
}
