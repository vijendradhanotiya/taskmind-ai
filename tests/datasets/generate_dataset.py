import json, random
from pathlib import Path
from itertools import product

random.seed(42)
OUT = Path(__file__).parent / "prompts_1000.jsonl"

NAMES = ["Rohan","Priya","Arpit","Neha","Shiv","Agrim","Vijendra","Karan","Rahul",
         "Ankit","Sahil","Meera","Pooja","Divya","Siddharth","Aditya","Gaurav",
         "Sumit","Amit","Riya","Dev","Nisha","Himanshu","Jatin","Ankur","Tarun"]

FEATURES = ["login page","payment module","dashboard","auth service","checkout flow",
            "search feature","notification service","onboarding flow","admin panel",
            "API integration","mobile app","reports module","CI/CD pipeline",
            "DB migration","recommendation engine","profile settings","billing module"]

PERCENTS = [20,30,40,50,55,60,65,70,75,80,85,90]

records = []
uid = 1

def add(endpoint, domain, inp, expected=None):
    global uid
    records.append({
        "id": uid, "domain": domain, "endpoint": endpoint,
        "input": inp, "expected_intent": expected
    })
    uid += 1

# ── TASK_ASSIGN (130) ────────────────────────────────────────────────────
assign_templates = [
    "@{name} please complete the {feature} by tomorrow",
    "@{name} fix the {feature} before EOD",
    "@{name} take ownership of the {feature} this sprint",
    "@{name} review the PR for {feature}",
    "@{name} the {feature} is pending from your end",
    "@{name} deploy the {feature} to staging today",
    "@{name} write unit tests for the {feature}",
    "@{name} {feature} ka kaam complete karo ASAP",
    "@{name} handle the client feedback on {feature}",
    "@{name} please review and merge the {feature} PR",
    "@{name} {feature} me bug fix karo, P0 issue hai",
    "@{name} update the documentation for {feature}",
]
for tpl in assign_templates:
    for name in random.sample(NAMES, 5):
        feat = random.choice(FEATURES)
        add("/v1/classify","task_assign", tpl.format(name=name, feature=feat), "TASK_ASSIGN")
        if uid > 131: break
    if uid > 131: break
while uid <= 130:
    n, f = random.choice(NAMES), random.choice(FEATURES)
    add("/v1/classify","task_assign", f"@{n} please complete the {f} ASAP","TASK_ASSIGN")

# ── TASK_DONE (90) ───────────────────────────────────────────────────────
done_templates = [
    "{feature} is done, pushed to staging",
    "completed {feature}, all tests passing",
    "{feature} shipped and live in production",
    "done bhai, {feature} merged to main",
    "finished {feature}, moving to next task",
    "PR merged for {feature}, ready for QA",
    "{feature} hotfix deployed to prod",
    "tested {feature}, no issues found",
    "{feature} feature branch merged successfully",
    "LGTM on {feature}, approved and merged",
]
for tpl in done_templates:
    for _ in range(9):
        add("/v1/classify","task_done", tpl.format(feature=random.choice(FEATURES)), "TASK_DONE")

# ── TASK_UPDATE (80) ─────────────────────────────────────────────────────
for feat in FEATURES:
    pct = random.choice(PERCENTS)
    add("/v1/classify","task_update", f"{feat} {pct}% done", "TASK_UPDATE")
update_phrases = [
    "working on {feature}, should be done by EOD",
    "{feature} in progress, about {pct}% complete",
    "still working on {feature}, need one more day",
    "{feature} ke baad done karunga, {pct}% ho gaya",
    "{feature} almost done, just testing left",
    "started {feature}, halfway through",
    "{feature} coding done, writing tests now",
    "{feature} deployed to staging, waiting for QA",
    "PR is up for {feature}, waiting for review",
    "{feature} {pct}% complete, on track for Friday",
]
for tpl in update_phrases:
    add("/v1/classify","task_update",
        tpl.format(feature=random.choice(FEATURES), pct=random.choice(PERCENTS)),
        "TASK_UPDATE")
for _ in range(80 - uid + 221):
    if uid > 300: break
    add("/v1/classify","task_update",
        f"{random.choice(FEATURES)} {random.choice(PERCENTS)}% ho gaya", "TASK_UPDATE")

# ── TASK_BLOCKED (60) ────────────────────────────────────────────────────
blocked = [
    "blocked on {feature}, waiting for credentials from vendor",
    "{feature} is broken, can't proceed",
    "getting 500 error on {feature}",
    "CI is failing, can't deploy {feature}",
    "{feature} stuck due to missing env variables",
    "can't start {feature} without product spec",
    "{feature} build failing with dependency conflict",
    "waiting for DevOps to unblock {feature}",
    "third party API down, {feature} integration stuck",
    "DB migration failing, blocking {feature} release",
    "staging env misconfigured, all {feature} tests failing",
    "getting OOM error in {feature} service",
]
for tpl in blocked:
    for _ in range(5):
        add("/v1/classify","task_blocked",
            tpl.format(feature=random.choice(FEATURES)), "TASK_BLOCKED")
        if uid > 360: break
    if uid > 360: break

# ── PROGRESS_NOTE (50) ───────────────────────────────────────────────────
notes = [
    "standup: worked on {feature} yesterday, continuing today",
    "EOD update: {feature} done, {feature2} in progress",
    "sprint at {pct}%, {feature} on track",
    "prod incident: {feature} was down for 10 mins, fixed",
    "weekly update: {feature} shipped, starting {feature2}",
    "deploy complete, {feature} is live, monitoring",
    "postmortem: {feature} outage caused by config change",
    "load test: {feature} stable at 800 rps",
    "A/B test result on {feature}: variant B wins by 12%",
    "release v2.1 scheduled Thursday, {feature} frozen",
]
for tpl in notes:
    for _ in range(5):
        add("/v1/classify","progress_note",
            tpl.format(feature=random.choice(FEATURES),
                       feature2=random.choice(FEATURES),
                       pct=random.choice(PERCENTS)), "PROGRESS_NOTE")
        if uid > 410: break
    if uid > 410: break

# ── GENERAL_MESSAGE (90) ─────────────────────────────────────────────────
general = [
    "good morning team!", "okay noted", "thanks bhai", "roger that",
    "sounds good to me", "hahaha", "will discuss in the call",
    "ping me when free", "good night everyone", "happy friday!",
    "who's joining the retro?", "zoom link?", "joining in 5 mins",
    "anyone free for a quick sync?", "link not working, reshare?",
    "congrats on the promotion!", "kal chutti hai", "LOL 😂",
    "what time is the client call?", "please add me to the invite",
    "where is the postman collection?", "thanks for the review!",
    "noted, will follow up", "same here", "apologies for the delay",
    "I'll be OOO tomorrow", "have a good weekend!", "back from lunch",
    "ek baar check karo", "bhai chai pi ke aao",
    "anyone on call this weekend?", "please don't push to main directly",
    "who reviewed this PR?", "koi nahi bhai ho jayega",
    "meeting rescheduled to 4pm", "is the retro still at 5pm?",
    "please test on iOS and Android", "glad that's resolved",
    "I agree with Rohan's approach", "let's discuss in sprint planning",
    "checking in — everyone aligned?", "haan bilkul",
    "thoda rush ho gaya aaj", "kal discuss karte hain",
    "who is the DRI for this?", "can someone share the figma link?",
    "all good, no action needed", "sent creds to your email",
    "I pushed a draft PR for early feedback", "great sprint everyone!",
    "the slides are in shared drive", "happy to pair program if needed",
    "interesting, I didn't know that", "we're a bit over budget",
    "let's keep this in the channel", "I'll push fix in 10 mins",
    "any thoughts on this approach?", "who leads the demo?",
    "please update your timesheets", "team lunch at 1pm tomorrow",
    "can we get access to staging?", "waiting for meeting to start",
    "what's the release checklist?", "is the office open Monday?",
    "will DM you the credentials", "can we push release to tomorrow?",
    "how is everyone doing?", "awesome work 🔥",
    "see you all in standup", "bhai kaafi solid kaam kiya tune",
    "please add comments, code is unclear", "sorry for the noise",
    "who owns the billing module?", "where are the API docs?",
    "ok I'll check", "please add me to the meeting",
    "that makes sense, thanks!", "great job on the delivery!",
    "anyone available for a quick call?", "noted!",
    "I had a look, approach seems solid", "good question, let me think",
    "retro highlights: need better staging comms",
    "sprint review done, client is happy",
    "team: Rohan on leave, Priya covering",
    "I'll follow up async",
    "please review before EOD",
    "added to backlog for next sprint",
    "confirmed, no blockers on my end",
    "update shared in the doc",
]
for msg in general[:90]:
    add("/v1/classify","general_message", msg, "GENERAL_MESSAGE")

# ── CHAT COMPLETIONS (250) ───────────────────────────────────────────────
ml_qs = [
    "What is LoRA in simple terms?","What does PEFT stand for?",
    "Explain fine-tuning vs pre-training","What is a transformer model?",
    "What is the attention mechanism?","What is overfitting?",
    "What is gradient descent?","What is a loss function?",
    "What is token accuracy?","Explain training loss vs eval loss",
    "Why does batch size matter?","How do you choose a learning rate?",
    "What is SFT training?","What are warmup steps?",
    "What is the HuggingFace Hub?","What is model quantization?",
    "What is MPS on Apple Silicon?","Explain RLHF simply",
    "Difference between BERT and GPT","What is top_p sampling?",
    "How does temperature affect output?","What is a tokenizer?",
    "What is a training checkpoint?","What is transfer learning?",
    "What is an embedding in NLP?","Why use adapters over full fine-tuning?",
    "What is TRL for LLM training?","What is a training epoch?",
    "Purpose of a validation set","What is cross-entropy loss?",
    "What is beam search?","What is model hallucination?",
    "What are activation functions?","What is dropout?",
    "What is layer normalisation?","RNN vs Transformer differences",
    "What is a vector database?","What is RAG for LLMs?",
    "What is the context window?","What is instruction tuning?",
    "What is early stopping?","What is the vanishing gradient problem?",
    "What is weight decay?","Explain cosine LR schedule",
    "What is a base model vs chat model?","What are BOS and EOS tokens?",
    "What is knowledge distillation?","What is flash attention?",
    "What is nucleus sampling?","What is perplexity as a metric?",
]
coding_qs = [
    "List vs tuple in Python","What is a Python decorator?",
    "What is async/await?","What does __init__ do?",
    "What is a REST API?","Why is FastAPI fast?",
    "What is Docker?","SQL vs NoSQL databases",
    "What is a Python virtual environment?","Git rebase vs merge",
    "What is a microservices architecture?","What is a webhook?",
    "What is idempotency in APIs?","What does a load balancer do?",
    "What is Redis used for?","What is a race condition?",
    "What is CI/CD?","What is Kubernetes?",
    "Difference between TCP and UDP","What is a message queue?",
    "What is OAuth 2.0?","What is a JWT token?",
    "What is API rate limiting?","Authentication vs authorisation",
    "What is caching?","What is eventual consistency?",
    "What is a CDN?","What is a design pattern?",
    "What is TDD?","Unit tests vs integration tests",
    "What is horizontal vs vertical scaling?","What is a DB index?",
    "What is a database deadlock?","What is an ORM?",
    "What is the CAP theorem?","What is dependency injection?",
    "What is binary search?","What is Big O notation?",
    "What is memoization?","What is Pydantic used for?",
    "What is GraphQL vs REST?","What is serverless computing?",
    "What is Infrastructure as Code?","What is a hash map?",
    "What is a Python generator?","What is type hinting?",
    "What is a shallow vs deep copy?","What is recursion?",
    "What is the singleton pattern?","What is a design pattern?",
]
general_qs = [
    "What is the capital of Japan?","Who invented the telephone?",
    "What is the speed of light?","What is photosynthesis?",
    "What is the Pythagorean theorem?","Who walked on the moon first?",
    "What is blockchain technology?","What is the greenhouse effect?",
    "What is GDP?","What is inflation?",
    "What is compound interest?","What is venture capital?",
    "What is a startup unicorn?","What is product-market fit?",
    "What is Agile methodology?","What is design thinking?",
    "What is the difference between UI and UX?","What is A/B testing?",
    "What is net promoter score?","What is customer lifetime value?",
    "What is GDPR?","What is two-factor authentication?",
    "What is end-to-end encryption?","What is open source software?",
    "SaaS vs PaaS vs IaaS","What is edge computing?",
    "What is 5G technology?","What is quantum computing?",
    "What is augmented reality?","What is Web3?",
    "What is an NFT?","Difference between ML and AI",
    "What is data science?","What is big data?",
    "What is the Turing test?","What is Moore's law?",
    "What is Y Combinator?","What is a startup pivot?",
    "What does MVP mean in product?","What is SCRUM?",
    "What is an OKR?","What is the Pareto principle?",
    "What is first principles thinking?","Who founded Apple?",
    "What is the stock market?","What is a bull vs bear market?",
    "What is supply and demand?","What is a monopoly?",
    "What is brand loyalty?","What is customer acquisition cost?",
]
career_qs = [
    "How to prepare for an ML interview?",
    "What should a junior ML engineer's resume include?",
    "How to negotiate salary as an engineer?",
    "How to transition from backend to ML engineering?",
    "What projects help get an AI job?",
    "How important is a HuggingFace profile for ML jobs?",
    "How do I get a remote US tech job from India?",
    "What is a take-home ML assignment?",
    "How to write a cold email to a recruiter?",
    "What is the STAR method for interviews?",
    "Is LeetCode important for ML roles?",
    "Should I do a Masters or go to industry for ML?",
    "What is the best way to showcase a fine-tuning project?",
    "How to build a strong GitHub portfolio?",
    "What is freelance ML consulting?",
    "What skills does an LLM engineer need?",
    "How to write a technical blog post for visibility?",
    "What is the difference between MLE and data scientist?",
    "How to find ML internships?",
    "How to get promoted in a tech company?",
    "What is a typical ML team structure?",
    "How to stay updated with ML research?",
    "What certifications help in ML career?",
    "Is Kaggle important for ML jobs?",
    "How to give a good ML system design interview?",
    "What is the best way to learn PyTorch?",
    "How to contribute to open source ML projects?",
    "What is MLOps and why does it matter?",
    "How to build a personal brand as an ML engineer?",
    "What is the glassdoor salary for ML engineers?",
]
productivity_qs = [
    "What is the Pomodoro technique?",
    "How to stay focused while working remotely?",
    "What are the best tools for async team communication?",
    "What is deep work and how to practice it?",
    "How to manage multiple projects at once?",
    "What is time blocking?","What is the Eisenhower matrix?",
    "How to run an effective sprint retrospective?",
    "What are good habits for software engineers?",
    "How to do a weekly review as a developer?",
    "What is personal knowledge management?",
    "What is the best note-taking system for engineers?",
    "How to avoid burnout in tech?",
    "How to give constructive code review feedback?",
    "What is the best way to document a codebase?",
    "How to estimate tasks accurately in a sprint?",
    "What is the difference between urgent and important?",
    "How to improve developer productivity?",
    "How to conduct a technical interview as an interviewer?",
    "What are the best books for software engineers?",
]

for q in ml_qs[:50]: add("/v1/chat/completions","ml_ai", q)
for q in coding_qs[:50]: add("/v1/chat/completions","coding", q)
for q in general_qs[:50]: add("/v1/chat/completions","general_knowledge", q)
for q in career_qs[:30]: add("/v1/chat/completions","career", q)
for q in productivity_qs[:20]: add("/v1/chat/completions","productivity", q)

# ── COMPLETIONS (250) ────────────────────────────────────────────────────
ml_starts = [
    "LoRA stands for","Fine-tuning a model means",
    "The purpose of a tokenizer is","The attention mechanism works by",
    "A transformer model is","Gradient descent is an algorithm that",
    "The loss function measures","Overfitting occurs when",
    "A training epoch is","Model quantization reduces",
    "The learning rate controls","A LoRA adapter adds",
    "Warmup steps help the optimizer","Token accuracy measures",
    "The HuggingFace library allows developers to",
    "An embedding vector represents","The softmax function converts",
    "RLHF stands for","A checkpoint saves",
    "The eval dataset is used to",
    "Apple Silicon uses MPS which stands for",
    "The base model contains","Transfer learning allows",
    "Batch normalisation helps the model","The context window of an LLM is",
    "Knowledge distillation is","A vector database stores",
    "RAG combines","Instruction tuning teaches the model to",
    "Top-p sampling selects",
]
coding_starts = [
    "Python is a programming language that","FastAPI is a Python framework for",
    "Docker containers help in","A REST API endpoint accepts",
    "Redis is an in-memory store used for","A database index speeds up",
    "OAuth 2.0 is a protocol for","A JWT token contains",
    "Kubernetes is used to","A message queue allows",
    "Rate limiting prevents","Caching improves performance by",
    "A load balancer distributes","Microservices architecture splits",
    "CI/CD automates","A webhook is triggered when",
    "An ORM maps","The CAP theorem states that",
    "A hash map stores","Binary search works by",
    "A Python decorator wraps","Async/await allows",
    "GraphQL differs from REST because","Serverless computing removes",
    "A virtual environment isolates","Type hinting in Python allows",
    "A generator in Python yields","Memoization caches",
    "The singleton pattern ensures","Dependency injection provides",
]
general_starts = [
    "The capital of France is","Machine learning is a branch of",
    "The greenhouse effect occurs when","Compound interest grows because",
    "The Pythagorean theorem states","Blockchain technology works by",
    "GDP measures","Inflation is caused by",
    "Venture capital is a form of","Agile methodology focuses on",
    "A/B testing compares","End-to-end encryption ensures",
    "Open source software allows","The Turing test evaluates",
    "Moore's law predicts","Quantum computing uses",
    "Product-market fit means","A startup pivot is",
    "The MVP in product development is","SCRUM is a framework for",
    "OKRs help teams by","The Pareto principle states",
    "First principles thinking involves","A CDN improves performance by",
    "5G technology enables","Edge computing processes data",
    "NFTs are unique because","Web3 is built on",
    "The stock market allows","Supply and demand determines",
]
business_starts = [
    "Customer lifetime value is calculated by",
    "Net promoter score measures","A SaaS product is",
    "Product-led growth means","Brand loyalty is built by",
    "Customer acquisition cost is",
    "A freemium model offers",
    "A go-to-market strategy defines",
    "Churn rate measures","Monthly recurring revenue is",
    "A competitive moat is","Unit economics describe",
    "Product roadmap prioritises","User retention is improved by",
    "The north star metric for a product is",
    "A growth hacker focuses on","Market segmentation divides",
    "A business model canvas has",
    "The rule of 40 for SaaS states",
    "Technical debt accumulates when",
]

for s in ml_starts:    add("/v1/completions","ml_completion", s)
for s in coding_starts: add("/v1/completions","coding_completion", s)
for s in general_starts: add("/v1/completions","general_completion", s)
for s in business_starts: add("/v1/completions","business_completion", s)

total_needed = 1000
while len(records) < total_needed:
    add("/v1/completions","general_completion",
        f"{random.choice(FEATURES)} in software engineering means")

records = records[:1000]
random.shuffle(records)
for i,r in enumerate(records):
    r["id"] = i + 1

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT,"w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

from collections import Counter
domains = Counter(r["domain"] for r in records)
endpoints = Counter(r["endpoint"] for r in records)
intents = Counter(r["expected_intent"] for r in records if r["expected_intent"])
print(f"Generated {len(records)} prompts → {OUT}")
print("\nBy endpoint:")
for k,v in endpoints.items(): print(f"  {k:<30} {v}")
print("\nBy domain (top 10):")
for k,v in domains.most_common(10): print(f"  {k:<30} {v}")
print("\nBy expected intent (classify only):")
for k,v in intents.items(): print(f"  {k:<20} {v}")
