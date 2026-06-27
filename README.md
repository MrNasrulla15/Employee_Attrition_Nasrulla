# Employee Attrition Prediction

A data science case study that asks a simple but expensive question: *can we tell, in advance, which employees are likely to quit?*

Using HR records for 1,470 employees, I built a machine learning model that flags at-risk employees before they hand in their resignation — and I turn those findings into recommendations HR can actually act on.

## Why this matters

Losing an employee isn't cheap. Between recruiting, onboarding, and the productivity dip while a replacement gets up to speed, turnover can cost half to two times that person's annual salary. Most companies only find out someone's unhappy after they've already accepted another offer — by then, it's too late to do anything but say goodbye.

The goal here is to shift that timeline earlier: identify the warning signs while there's still time to act on them.

## What's in this folder

| File | What it is |
|---|---|
| `analysis.ipynb` | The full analysis, start to finish — cleaning, exploration, modeling, evaluation, and conclusions, fully executed |
| `summary.docx` | A one-page summary written for an HR Director, no technical jargon |
| `charts/` | The key charts from the notebook, exported as high-resolution PNGs |

## What the data actually showed

A few findings stood out enough that they're worth knowing even before opening the notebook:

- **Overtime is the loudest warning sign.** Employees working overtime leave at **30.5%**, compared to **10.4%** for everyone else — nearly a 3x difference, and one of the few drivers HR can directly control.
- **One job, in particular, is in trouble.** Sales Representatives leave at **39.8%** — by far the highest of any role, and almost double the next closest one.
- **The danger zone is early.** Employees who quit had been there a median of **3 years**; those who stayed, **6 years**. Whatever's going wrong tends to go wrong early.
- **Money is part of the story, not the whole story.** People who left earned a median of **$3,202/month** versus **$5,204/month** for those who stayed — a real gap, but not big enough on its own to explain everything, since some well-paid roles (Managers, for instance) barely lose anyone.
- **A bad work-life balance rating roughly doubles the risk.** Employees who rated theirs as "Bad" left at **31.3%**, versus 14–18% for every better rating.

## How the model was built

The notebook walks through a fairly standard, careful pipeline:

1. **Explore first.** I look at the data's shape, types, and how balanced the target is (only ~16% of employees actually left, so this isn't a 50/50 split).
2. **Clean it up.** I drop columns that carry no real signal (like `EmployeeNumber`, which is just an ID), encode the target as 0/1, and one-hot encode categorical fields like department and job role.
3. **Split before scaling.** The train/test split happens *before* any scaling, and the scaler is fit only on the training data — a small detail, but skipping it is one of the most common ways people accidentally let the test set "leak" into training.
4. **Try a few models.** I train Logistic Regression, Random Forest, and Gradient Boosting and compare them fairly on the exact same data split.
5. **Judge them honestly.** Rather than picking whichever model scored the highest accuracy, I weigh Recall and F1 more heavily — because for HR, missing someone who was actually about to quit is a far more expensive mistake than flagging someone who was never at risk.

**Logistic Regression ended up winning** — not the fanciest model, but the one that struck the best balance:

| Metric | Score |
|---|---|
| Accuracy | 75.5% |
| Recall (catches employees who actually leave) | 66.0% |
| ROC AUC | 0.80 |

## Running it yourself

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook analysis.ipynb
```

Run it top to bottom — it's self-contained, seeded for reproducibility (`random_state=42`), and will regenerate every chart into `charts/` on its own.

## Worth keeping in mind

This model can tell you what's *associated* with people leaving — it can't prove that fixing one of those things (say, cutting back on overtime) will reduce attrition by some exact percentage. Correlation isn't causation, even when the pattern is strong. Before rolling any of these recommendations out company-wide, it's worth testing them on a smaller team first and seeing what actually moves the needle.

## Built with

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`
