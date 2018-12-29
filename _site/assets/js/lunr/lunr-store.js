var store = [{
        "title": "Advanced Usage",
        "excerpt":"How to Define a Customized Distribution Turing.jl supports the use of distributions from the Distributions.jl package. By extension it also supports the use of customized distributions, by defining them as subtypes of Distribution type of the Distributions.jl package, as well as corresponding functions. Below shows a workflow of how to...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/advanced/",
        "teaser":null},{
        "title": "Library",
        "excerpt":"Modelling # Turing.@model — Macro. @model(name, fbody)Macro to specify a probabilistic model. Example: @model Gaussian(x) = begin s ~ InverseGamma(2,3) m ~ Normal(0,sqrt.(s)) for i in 1:length(x) x[i] ~ Normal(m, sqrt.(s)) end return (s, m)endCompiler design: sample(fname(x,y), sampler). fname(x=nothing,y=nothing; compiler=compiler) = begin ex = quote # Pour in kwargs for those...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/library/",
        "teaser":null},{
        "title": "Contributing",
        "excerpt":"Turing is an open source project. If you feel that you have some relevant skills and are interested in contributing, then please do get in touch. You can contribute by opening issues on GitHub or implementing things yourself and making a pull request. We would also appreciate example models written...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/contributing/",
        "teaser":null},{
        "title": "Style Guide",
        "excerpt":"This style guide is adapted from Invenia’s style guide. We would like to thank them for allowing us to access and use it. Please don’t let not having read it stop you from contributing to Turing! No one will be annoyed if you open a PR whose style doesn’t follow...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/style-guide/",
        "teaser":null},{
        "title": "Getting Started",
        "excerpt":"Installation To use Turing, you need to install Julia first and then install Turing. Install Julia You will need to install Julia 1.0 or greater, which you can get from the official Julia website. Install Turing.jl Turing is an officially registered Julia package, so the following will install a stable...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/get-started/",
        "teaser":null},{
        "title": "Guide",
        "excerpt":"Basics Introduction A probabilistic program is Julia code wrapped in a @model macro. It can use arbitrary Julia code, but to ensure correctness of inference it should not have external effects or modify global state. Stack-allocated variables are safe, but mutable heap-allocated objects may lead to subtle bugs when using...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/guide/",
        "teaser":null},{
        "title": "Turing Documentation",
        "excerpt":"Welcome to the documentation for Turing 0.5. Introduction Turing is a universal probabilistic programming language with an intuitive modelling interface, composable probabilistic inference and computational scalability. Turing provides Hamiltonian Monte Carlo (HMC) and particle MCMC sampling algorithms for complex posterior distributions (e.g. those involving discrete variables and stochastic control flows)....","categories": [],
        "tags": [],
        "url": "http://localhost:4000/docs/",
        "teaser":null},{
        "title": "Tutorials",
        "excerpt":"This section contains tutorials on how to implement common models in Turing. If you prefer to have an interactive Jupyter notebook, please fork or download the TuringTutorials repository. A list of all the tutorials available can be found to the right. The introduction tutorial contains an introduction to coin flipping...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/tutorials/",
        "teaser":null}]
