# TODO

- [ ] Milestone 1
    - [ ] tests
        - [x] setup tests
        - [x] write all tests
        - [x] fix TODOs with Yuqi
        - [ ] add tissuemap test
    - [ ] modularize: pl (plot?), tl (tools), hp (helper), pp (preprocessing)
        - [x] initial modularization
        - [x] **Yuqi:** what is the difference between tools and helpers?
            - helpers are "minor" function ... potentially arbitrary
            - [ ] for now: any helper that is not used standalone move to tools
        - [x] **Yuqi:** segmentation stuff should be moved to different files?
            - modularization is appreciated :)
        - [x] **Yuqi:** some stuff plots even though it is not called `pl`!
            - optimally plotting code should not be in tool function
            - or split into helpers/tools and plots
    - [ ] fix environment conflicts
    - [ ] docs
        - [x] move docs to markdown
        - [ ] tutorial notebooks
    - [ ] timings of notebooks
        - [ ] based on what data?
    - [ ] optional
        - [ ] make files smaller; they are way too long (modularize)
    - [ ] **Yuqi**
        - [x] send qpdiff (optimally downsized)
        - [ ] tissue map

- [ ] MAYBE MANAGE THIS IN GITHUB ISSUES?
- [ ] Features implementationssomehow
        - [ ] Cellseg implementation @Yuqi (Linux version??)
        - [ ] GPU-enabled clustering @Tim
        - [ ] Fix the filtering step so that if one condition's p-value is
                below the threshold @Yuqi
        - [ ] Fix the plotting function so that it indicates which p-value
                are not below the selected threshold @Yuqi
