---
title: "Snowflurry/en"
url: "https://docs.alliancecan.ca/wiki/Snowflurry/en"
category: "General"
last_modified: "2024-10-24T20:38:45Z"
page_id: 26343
display_title: "Snowflurry"
---

`<languages />`{=html} [Snowflurry](https://github.com/SnowflurrySDK/Snowflurry.jl/) is an open-source quantum computing library developed in [Julia](https://docs.alliancecan.ca/Julia "Julia"){.wikilink} by [Anyon Systems](https://anyonsys.com/) that allows you to build, simulate, and run quantum circuits. A related library called [SnowflurryPlots](https://github.com/SnowflurrySDK/SnowflurryPlots.jl/) allows you to visualize the simulation results in a bar chart. Useful to explore quantum computing, its features are described in the [documentation](https://snowflurrysdk.github.io/Snowflurry.jl/dev/index.html) and the [installation guide is available on the GitHub page](https://github.com/SnowflurrySDK/Snowflurry.jl). Like the [PennyLane](https://docs.alliancecan.ca/PennyLane/en "PennyLane"){.wikilink} library, Snowflurry can be used to run quantum circuits on the [MonarQ](https://docs.alliancecan.ca/MonarQ/en "MonarQ"){.wikilink} quantum computer.

## Installation

The quantum computer simulator with [Snowflurry](https://github.com/SnowflurrySDK/Snowflurry.jl) is available on all of our clusters. The [Julia](https://julialang.org/) programming language must be loaded before accessing Snowflurry. `<includeonly>`{=html}

<div class="floatright">

![](Question.png "Question.png"){width="40"}

</div>
<div class="command">

} }}\|lang=}}}

</div>

`</includeonly>`{=html}`<noinclude>`{=html}

`</noinclude>`{=html} The Julia programming interface is then called and the Snowflurry quantum library is loaded (in about 5-10 minutes) with the commands `<includeonly>`{=html}

<div class="floatright">

![](Question.png "Question.png"){width="40"}

</div>
<div class="command">

} }}\|lang=}}}

</div>

`</includeonly>`{=html}`<noinclude>`{=html}

`</noinclude>`{=html} Quantum logic gates and commands are described in the [Snowflurry documentation](https://snowflurrysdk.github.io/Snowflurry.jl/dev/).

## Use case: Bell states {#use_case_bell_states}

Bell states are maximally entangled two-qubit states. They are simple examples of two quantum phenomena: superposition and entanglement. The [Snowflurry](https://github.com/SnowflurrySDK/Snowflurry.jl/) library allows you to construct the first Bell state as follows: `<noinclude>`{=html}

`</noinclude>`{=html} In the above code section, the Hadamard gate creates an equal superposition of \|0⟩ and \|1⟩ on the first qubit while the CNOT gate (controlled X gate) creates an entanglement between the two qubits. We find an equal superposition of states \|00⟩ and \|11⟩, which is the first Bell state. The `simulate` function allows us to simulate the exact state of the system. `<noinclude>`{=html}

` julia> state = simulate(circuit)`\
` julia> print(state)   `\
` 4-element Ket{ComplexF64}:`\
` 0.7071067811865475 + 0.0im`\
` 0.0 + 0.0im`\
` 0.0 + 0.0im`\
` 0.7071067811865475 + 0.0im`

`</noinclude>`{=html}

The `readout` operation lets you specify which qubits will be measured. The `plot_histogram` function from the SnowflurryPlots library allows you to visualize the results. `<noinclude>`{=html}

`</noinclude>`{=html} ![](Bell_Graph.png "Bell_Graph.png")
