module TestHelpers
    export optimizer_factory
    using Clarabel
    function optimizer_factory()
        return Clarabel.Optimizer
    end
end
