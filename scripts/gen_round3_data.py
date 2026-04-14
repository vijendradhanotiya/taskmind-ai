"""
gen_round3_data.py  --  Generate 400+ targeted training examples that fix the
3 critical failure patterns found in the 500-sample test:

  1. PROGRESS_NOTE  2% accuracy  -> generate 200 broadcast/standup/EOD examples
  2. TASK_ASSIGN   33% accuracy  -> generate 120 "@name please complete" examples
  3. TASK_UPDATE   51% accuracy  -> generate 80  Hindi "ho gaya" + blocker examples

Output: data/round3/train_generated.jsonl
"""

import json
import random
from pathlib import Path

random.seed(42)

SYSTEM = (
    "You are TaskMind. Read the team WhatsApp message and return ONLY a JSON "
    "object with these exact fields: intent (TASK_ASSIGN / TASK_DONE / "
    "TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project, "
    "title, deadline, priority, progressPercent. Use null for unknown fields.\n\n"
    "Critical distinctions:\n"
    "  TASK_ASSIGN    = giving a task TO someone (@name + imperative verb, future action)\n"
    "  TASK_DONE      = work already finished (merged, shipped, deployed, completed, tested)\n"
    "  TASK_UPDATE    = partial progress or blocker (X% done/ho gaya, stuck, waiting, can't start)\n"
    "  PROGRESS_NOTE  = team broadcast: standup, EOD update, weekly/sprint report, load test,\n"
    "                   A/B test result, postmortem, retro highlights, release schedule\n"
    "  GENERAL_MESSAGE = social, greetings, reactions, logistics questions, acknowledgements"
)

NAMES = ["Arpit", "Neha", "Rohan", "Priya", "Karan", "Divya", "Shiv", "Rahul",
         "Meera", "Vijendra", "Tarun", "Jatin", "Riya", "Ankur", "Sumit",
         "Pooja", "Gaurav", "Dev", "Siddharth", "Aditya", "Himanshu", "Amit"]

FEATURES = ["dashboard", "login page", "payment module", "auth service", "admin panel",
            "search feature", "onboarding flow", "notification service", "CI/CD pipeline",
            "API integration", "DB migration", "mobile app", "reports module",
            "billing module", "checkout flow", "recommendation engine", "profile settings"]

PERCENTS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]


def make_entry(message, intent, assignee=None, title=None, progress=None):
    prompt = "### System:\n" + SYSTEM + "\n\n### Message:\n" + message + "\n\n### Response:\n"
    completion = json.dumps({
        "intent": intent,
        "assigneeName": assignee,
        "project": None,
        "title": title,
        "deadline": None,
        "priority": None,
        "progressPercent": progress,
    })
    return {"prompt": prompt, "completion": completion, "text": prompt + completion}


def gen_progress_notes():
    examples = []

    # standup updates
    for feat in FEATURES:
        pct = random.choice(PERCENTS)
        examples.append(make_entry(
            f"standup: worked on {feat} yesterday, {pct}% done, continuing today",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"standup update — {feat} at {pct}%, should finish by EOD",
            "PROGRESS_NOTE"
        ))

    # EOD updates
    for i in range(len(FEATURES)):
        f1 = FEATURES[i]
        f2 = FEATURES[(i + 1) % len(FEATURES)]
        examples.append(make_entry(
            f"EOD update: {f1} done, starting {f2} tomorrow",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"EOD: finished {f1}, {f2} kicks off in morning standup",
            "PROGRESS_NOTE"
        ))

    # weekly / sprint updates
    for feat in FEATURES:
        pct = random.choice([60, 65, 70, 75, 80])
        examples.append(make_entry(
            f"weekly update: {feat} shipped, starting {FEATURES[PERCENTS.index(pct) % len(FEATURES)]} next week",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"sprint at {pct}%, {feat} on track for Friday delivery",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"sprint review done — {feat} delivered, client approved",
            "PROGRESS_NOTE"
        ))

    # load test results
    for feat in FEATURES[:10]:
        rps = random.choice([500, 600, 800, 1000, 1200])
        examples.append(make_entry(
            f"load test: {feat} stable at {rps} rps, p99 latency under 200ms",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"load test result — {feat} handles {rps} concurrent users, no errors",
            "PROGRESS_NOTE"
        ))

    # A/B test results
    for feat in FEATURES[:8]:
        examples.append(make_entry(
            f"A/B test result on {feat}: variant B wins with +12% conversion",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"A/B test on {feat} concluded — control group performed better",
            "PROGRESS_NOTE"
        ))

    # postmortems
    for feat in FEATURES[:8]:
        examples.append(make_entry(
            f"postmortem: {feat} outage caused by a bad config deploy, fixed",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"postmortem done for {feat} incident — root cause: DB connection pool exhausted",
            "PROGRESS_NOTE"
        ))

    # retro highlights
    examples += [
        make_entry("retro highlights: need better staging coverage, adding more tests next sprint", "PROGRESS_NOTE"),
        make_entry("retro: team velocity improved, fewer bugs in QA this sprint", "PROGRESS_NOTE"),
        make_entry("sprint retro done — we shipped 8/10 stories, 2 carry forward", "PROGRESS_NOTE"),
        make_entry("retro action items: improve PR turnaround time and add pre-commit hooks", "PROGRESS_NOTE"),
        make_entry("retro note: deployment process needs simplification, agreed to automate", "PROGRESS_NOTE"),
    ]

    # release schedules
    for feat in random.sample(FEATURES, 8):
        examples.append(make_entry(
            f"release v2.1 scheduled Thursday — {feat} included, QA sign-off pending",
            "PROGRESS_NOTE"
        ))
        examples.append(make_entry(
            f"release plan updated: {feat} going out Friday, hold deploys till then",
            "PROGRESS_NOTE"
        ))

    # prod incidents
    for feat in FEATURES[:8]:
        examples.append(make_entry(
            f"prod incident: {feat} was down for 15 mins, rolled back, monitoring",
            "PROGRESS_NOTE"
        ))

    return examples


def gen_task_assign_complete():
    examples = []
    actions = ["please complete", "please finish", "ka kaam complete karo", "complete kar lo",
               "please wrap up", "finish karo", "please close out"]
    deadlines = ["ASAP", "by EOD", "by Friday", "this sprint", "by tomorrow", "before EOD"]

    for name in NAMES:
        for feat in FEATURES:
            action = random.choice(actions)
            deadline = random.choice(deadlines)
            msg = f"@{name} {action} the {feat} {deadline}"
            examples.append(make_entry(msg, "TASK_ASSIGN", assignee=name, title=feat))

    # ownership transfers
    for name in NAMES[:10]:
        for feat in random.sample(FEATURES, 3):
            examples.append(make_entry(
                f"@{name} take ownership of the {feat} task",
                "TASK_ASSIGN", assignee=name
            ))

    # review tasks
    for name in NAMES[:10]:
        for feat in random.sample(FEATURES, 3):
            examples.append(make_entry(
                f"@{name} please review and merge the {feat} PR",
                "TASK_ASSIGN", assignee=name
            ))

    return examples


def gen_task_update_hindi():
    examples = []
    hindi_variants = [
        "{feat} {pct}% ho gaya",
        "{feat} {pct}% complete ho gaya",
        "{feat} ka kaam {pct}% hua",
        "{feat} mein {pct}% progress",
        "abhi {feat} {pct}% hai",
        "{feat} {pct}% ho gayi hai",
    ]
    blockers = [
        "can't start {feat} without product specs",
        "can't proceed on {feat}, waiting for design approval",
        "{feat} blocked, credentials not shared yet",
        "stuck on {feat}, need access to staging",
        "{feat} pe kaam nahi ho raha, env setup pending",
        "waiting for DevOps to unblock {feat}",
    ]

    for feat in FEATURES:
        for pct in random.sample(PERCENTS, 5):
            template = random.choice(hindi_variants)
            msg = template.format(feat=feat, pct=pct)
            examples.append(make_entry(msg, "TASK_UPDATE", progress=pct))

    for feat in FEATURES:
        template = random.choice(blockers)
        msg = template.format(feat=feat)
        examples.append(make_entry(msg, "TASK_UPDATE"))

    return examples


def gen_general_message_guards():
    examples = []
    social = [
        "have a good weekend!", "happy friday!", "good night everyone",
        "great sprint everyone!", "congrats on the promotion!", "well done team!",
        "bhai kaafi solid kaam kiya tune", "awesome work on this!", "respect bhai",
        "zoom link?", "can someone share the figma link?", "who leads the demo?",
        "team lunch at 1pm tomorrow", "please add me to the invite",
        "please add me to the meeting", "please add me to the calendar invite",
        "roger that", "confirmed, no blockers on my end",
        "sent creds to your email", "I agree with the approach",
        "thanks for the review!", "great work on the delivery!",
        "I'll be OOO tomorrow", "joining in 5 mins", "back from lunch",
        "happy to pair program if needed", "can we get access to staging?",
        "who reviewed this PR?", "where are the API docs?",
        "is the retro still at 5pm?", "meeting rescheduled to 4pm",
        "please update your timesheets", "please add comments, code is unclear",
        "please review before EOD", "let's discuss in sprint planning",
        "let's keep this in the channel", "any thoughts on this approach?",
        "interesting, I didn't know that", "I had a look, approach seems solid",
        "ok I'll check", "will DM you the credentials",
        "update shared in the doc", "all good, no action needed",
        "who's joining the retro?", "anyone free for a quick sync?",
    ]
    for msg in social:
        examples.append(make_entry(msg, "GENERAL_MESSAGE"))
    return examples


def main():
    progress_note_ex = gen_progress_notes()
    task_assign_ex   = gen_task_assign_complete()
    task_update_ex   = gen_task_update_hindi()
    general_ex       = gen_general_message_guards()

    all_examples = progress_note_ex + task_assign_ex + task_update_ex + general_ex
    random.shuffle(all_examples)

    out = Path("data/round3/train_generated.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in all_examples:
            f.write(json.dumps(e) + "\n")

    from collections import Counter
    dist = Counter(json.loads(e["completion"])["intent"] for e in all_examples)
    print(f"Generated {len(all_examples)} examples -> {out}")
    print("Intent distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
