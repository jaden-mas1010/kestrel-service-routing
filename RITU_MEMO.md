# Decision memo | Ritu Deshpande, Head of D2C Operations

**Decision:** Pilot RouteDesk alongside the current bot; do not cancel the licence yet.

**The number:** The local classifier matched 93.7% of original routing labels on 2,165 later historical requests (137 mismatches). This is a test of *bot-label matching*, not evidence that it prevents incorrect transfers. The hidden-file score and live accuracy remain unknown.

**The rupees:** Replacing the vendor licence could avoid at most **₹3,20,000 gross annually**, or ~₹26,667 a month before costs. Each team transfer costs ₹305 and the associated extra contact averages ₹260, so reducing one avoidable misroute could save ~₹565, but no verified reduction is claimed yet. Actual net savings require staff-review effort, hosting, upkeep and errors to be measured. No paid model API is used.

**Next week:** Run a one-week shadow pilot. Log the recommended team, human correction, actual resolving team, number of transfers, and handling costs. Separately monitor warranty claims, installation paid-but-not-done messages, and purifier repair versus filter requests. Obtain a written go/no-go decision based on resolution outcomes and operational cost rather than historical label matching alone.

**Known risk:** Training labels are the original bot's initial assignments, including misroutes. RouteDesk may accurately reproduce those historical errors. Low-information tickets need human review.
