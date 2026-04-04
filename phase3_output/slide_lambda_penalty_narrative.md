# Slide: From λ penalties to physics

## Headline
**Math penalties (λ) are inefficient. What if we used physics instead?**

## The setup (say this out loud)
- Classical portfolio constraints (budget, concentration, “don’t pile into correlated risk”) get **squashed into a single unconstrained objective** by adding **quadratic penalty terms** weighted by **λ**.
- In the hackathon formulation, a budget like **∑_i w_i = B** becomes **λ(∑_i w_i − B)²** so feasible portfolios sit in **wells** of the energy landscape.

## The frustrating λ tradeoff
- **λ too large:** penalties dominate → the optimizer “forgets” returns and chases feasibility / trivial low-energy states.
- **λ too small:** constraints are weak → **infeasible** portfolios (wrong cardinality, wrong budget) look artificially good.

## Pivot line (your novelty hook)
- **Classical QUBO / penalty methods = hand-tuned λ** to fake constraints inside one objective.
- **Physical devices (Hamiltonian dynamics, hardware graphs, noise)** impose **interactions and locality** that are not the same as a spreadsheet penalty — the question is whether **physics** can implement **structure** more naturally than **λ whack-a-mole**.

## Optional one-liner for the next slide
“We still build **Q** — but the story moves from **tuning λ** to **engineering the right physical Hamiltonian and topology**.”
