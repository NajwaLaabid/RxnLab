# Migrating to Modal/CPU only platform
- still evaluating the merits of the idea. 
  - technical requirements: I expect to be able to run most models on cpu, but would like to connect modal as a backup to be called in specific situations (with some models in particular, in batch jobs, if too many users are using the platform and I need to speed up inference, of course with rate limits for modal)
  - pros: won't have to rely on csc's infrastructure which is tied to my aalto connection/contract, so def worth considering
  - cons: how much will I have to pay? is there a service that offers everything I need (hosting the platform, saving the feedback in a database I can extract later, controlled privacy etc)
  - decision: get a list of potential candidate platforms, understand how they work in details, and decide if the migration is worthwhile

# Interface
- remove references to scr and retroanalyze as separate platforms for now, the final product will likely include them as features where relevant
- change the tagline of the platform, and potentially the landing page too:
  - current description: 
    A retrosynthesis research platform
    ML-driven retrosynthesis, one reaction at a time.
    RxnLab is a research playground for testing, steering, and analyzing machine-learning models for single-step retrosynthesis. Built for chemists and ML researchers to explore what modern generative models can do at the reaction level.
  - my feedback:
    - not sure how much to highlight the 'researchiness' of the platform? can I advertise it as an open-sourced platform for running and evaluating ML models for retrosynthesis? (more product targeted language)
    - the layout is also a bit weird, but I haven't found an aesthetic I like yet, and dont have much background in UX design besides. I need help finding examples of similar platform to understand what standard UX practices exist and choose an aesthetic. Here is an aesthetic that caught my eye: https://startup-exploration-day.lovable.app/
    - landing page should have the name of the platform, a tag line, and a description of what it does (maybe what features are available?)

# Collecting UX/platform feedback
- this is separate from the chemical feedback I save in the platform, this is just for me to know how users feel about the UI/UX, how they feel about the features available and how they are presented, how they would use such a platform if available to them, etc
- Not sure yet which feedback I want to collect specifically so need to brainstorm this too. 
- Also unsure how to add it to the platform (could maybe add a description explaining it's in beta mode) so its intuitive to use and non-intrusive UI-wise. Simplest idea is a link to a google form embedded in the page.

# Done
## Inpainting
(no active issues)

## TODO [DONE]
- get diffalign samples to display on wsgi (get rid of dummy mols)
- integrate DiffAlign with syntheseus and add evaluation code with rxn-insight
- add options to: 1) apply full evaluation pipeline, 2) display precursors with levels of filtering, 3) add additional info next to the precursors
- selection highlights: now highlight bonds between selected atoms too, so
  implicit-carbon fragments show up visually
- selection summary: shows real fragment strings like "CC, C=O" per component
  instead of "N/M atoms of FULL_SMILES"
- fix /api/inpaint IndexError when sample_for_condition returns already-collapsed X