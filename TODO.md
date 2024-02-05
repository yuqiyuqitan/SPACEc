# TODO

- [ ] Milestone 2 (repository public)
    - [ ] docs
        - [ ] auto push docs to readthedocs
    - [ ] `README.md` add batches

- [ ] Milestone 1
    - [ ] fix environment conflicts
        - [ ] try to run tests based on yuqis minimal version `pip  --no-dependencies`
        - [ ] try to reproduce yuqi's minimal version (optimize)
            - look at minimal environment (`README.md`)
            - currently it looks like she somehow installs tensorflow 2.10
                and deepcell (which requires 2.8) still works
    - [ ] tests
        - [ ] fix tests on windows
        - [ ] add tissuemap test (not running yet)
    - [ ] Martin
        - [x] docs
            - [x] move docs to markdown
            - [x] tutorial notebooks
    - [ ] **Yuqi**
        - [x] send qpdiff (optimally downsized)
        - [ ] show Martin a working version of tissue map
        - [ ] cleanup jupyter notebooks (naming and headers)

- [ ] Features implementationssomehow
    - [ ] Cellseg implementation @Yuqi (Linux version??)
    - [ ] GPU-enabled clustering @Tim
    - [x] Fix the filtering step so that if one condition's p-value is
            below the threshold @Yuqi
    - [ ] Fix the plotting function so that it indicates which p-value
            are not below the selected threshold @Yuqi
