import json, random
from pathlib import Path

random.seed(42)

SAMPLES = {
  "instruction": {
    "easy": [
      {"prompt": "List the primary colors.", "completion": "The primary colors are red, blue, and yellow."},
      {"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."},
      {"prompt": "Convert 100 Fahrenheit to Celsius.", "completion": "100°F is approximately 37.8°C. Formula: (F - 32) × 5/9."},
      {"prompt": "Name three mammals.", "completion": "Three mammals are dogs, elephants, and dolphins."},
      {"prompt": "What does HTML stand for?", "completion": "HTML stands for HyperText Markup Language."},
      {"prompt": "How many days are in a leap year?", "completion": "A leap year has 366 days."},
      {"prompt": "What is the square root of 144?", "completion": "The square root of 144 is 12."},
      {"prompt": "Name the four seasons.", "completion": "The four seasons are spring, summer, autumn (fall), and winter."},
      {"prompt": "What is the chemical symbol for water?", "completion": "The chemical symbol for water is H₂O."},
      {"prompt": "List three types of renewable energy.", "completion": "Three types of renewable energy are solar, wind, and hydroelectric power."},
    ],
    "medium": [
      {"prompt": "Explain the steps to write a professional email.", "completion": "1. Use a clear subject line.\n2. Open with a polite greeting.\n3. State your purpose concisely in the first sentence.\n4. Provide necessary details in short paragraphs.\n5. Close with a call to action.\n6. End with a professional sign-off like 'Best regards'."},
      {"prompt": "What is machine learning and how does it differ from traditional programming?", "completion": "Machine learning is a method where systems learn patterns from data rather than following explicit rules. In traditional programming, a developer writes exact rules. In ML, the model infers rules from examples, making it better suited for complex tasks like image recognition or language understanding."},
      {"prompt": "Describe three strategies to improve focus while working from home.", "completion": "1. Time-blocking: Schedule specific tasks in fixed time slots to reduce decision fatigue.\n2. Environment separation: Dedicate a physical space solely for work to signal focus to your brain.\n3. Notification batching: Check messages at set intervals rather than responding in real time to protect deep work sessions."},
      {"prompt": "Explain the difference between a list and a tuple in Python.", "completion": "Both are ordered sequences in Python. A list is mutable — you can add, remove, or change elements after creation. A tuple is immutable — once created, it cannot be modified. Use tuples for fixed data (e.g., coordinates) and lists for data that changes (e.g., a shopping cart)."},
      {"prompt": "What are the key components of a healthy diet?", "completion": "A healthy diet includes: sufficient protein for muscle repair, complex carbohydrates for sustained energy, healthy fats (unsaturated) for brain function, fiber for digestion, vitamins and minerals from a variety of vegetables and fruits, and adequate hydration (6–8 glasses of water daily)."},
    ],
    "hard": [
      {"prompt": "You are a helpful assistant. A user asks: 'I need to improve team productivity without increasing headcount. What framework would you recommend and why?' Provide a structured response.", "completion": "I'd recommend the OKR (Objectives & Key Results) framework combined with async-first communication.\n\nWhy OKRs: They align individual work to company goals, reduce redundant effort, and surface blockers early. Each team sets 3–5 objectives with 2–3 measurable key results per quarter.\n\nWhy async-first: Meetings are the primary productivity drain. By defaulting to written updates, you reduce context-switching and return deep-work time to your team.\n\nImplementation steps:\n1. Set company-level OKRs top-down.\n2. Hold a 2-hour team workshop to define team OKRs.\n3. Weekly async check-in (written, 5 min per person).\n4. Monthly 30-min sync to unblock.\n\nExpected outcome: 15–25% productivity gain within 2 quarters."},
      {"prompt": "Describe how to design a REST API for a task management application. Include endpoints, methods, and response formats.", "completion": "A task management REST API should expose the following endpoints:\n\nGET /tasks — returns a paginated list of tasks\nPOST /tasks — creates a new task (body: {title, description, due_date, priority})\nGET /tasks/{id} — retrieves a single task\nPUT /tasks/{id} — full update of a task\nPATCH /tasks/{id} — partial update (e.g., mark complete)\nDELETE /tasks/{id} — soft-deletes a task\n\nAll responses return JSON with a consistent envelope: {data, error, meta}. Use HTTP status codes semantically: 200 OK, 201 Created, 400 Bad Request, 404 Not Found, 422 Unprocessable Entity. Include pagination metadata (page, per_page, total) on list endpoints."},
      {"prompt": "Explain the pros and cons of microservices architecture vs monolithic architecture for a startup with 5 engineers.", "completion": "For a 5-person startup, a monolith is almost always the right starting point.\n\nMonolith pros: faster development, simpler deployment, easier debugging, lower operational overhead. One codebase means no network latency between services and no distributed tracing complexity.\n\nMonolith cons: harder to scale individual components; tech debt accumulates; large teams can conflict on the same codebase.\n\nMicroservices pros: independent scaling, technology flexibility, isolated failures, team autonomy at scale.\n\nMicroservices cons: extreme operational complexity (service discovery, distributed tracing, network failures, data consistency). 5 engineers will spend more time on infra than product.\n\nRecommendation: Start with a well-structured modular monolith. Extract services only when a specific component has a clear, proven scaling bottleneck. Amazon, Shopify, and Stack Overflow ran monoliths profitably for years."},
    ]
  },
  "qa": {
    "easy": [
      {"prompt": "Q: Who invented the telephone?\nA:", "completion": "Alexander Graham Bell is credited with inventing the telephone in 1876."},
      {"prompt": "Q: What is the boiling point of water at sea level?\nA:", "completion": "Water boils at 100°C (212°F) at sea level under 1 atmosphere of pressure."},
      {"prompt": "Q: How many planets are in our solar system?\nA:", "completion": "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."},
      {"prompt": "Q: What is the largest ocean on Earth?\nA:", "completion": "The Pacific Ocean is the largest ocean, covering about 165 million square kilometers."},
      {"prompt": "Q: What year did World War II end?\nA:", "completion": "World War II ended in 1945 — in Europe on May 8 (V-E Day) and in the Pacific on September 2 (V-J Day)."},
      {"prompt": "Q: What is the speed of light?\nA:", "completion": "The speed of light in a vacuum is approximately 299,792 kilometers per second (about 186,000 miles per second)."},
      {"prompt": "Q: What is photosynthesis?\nA:", "completion": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."},
      {"prompt": "Q: What does CPU stand for?\nA:", "completion": "CPU stands for Central Processing Unit, the primary component of a computer that executes instructions."},
    ],
    "medium": [
      {"prompt": "Q: Why does the sky appear blue during the day but red/orange at sunset?\nA:", "completion": "The sky is blue because of Rayleigh scattering. Sunlight contains all wavelengths; shorter blue wavelengths scatter more easily off atmospheric gas molecules, flooding the sky with blue light. At sunset, light travels through more atmosphere. Most blue is scattered away before reaching your eyes, leaving the longer red and orange wavelengths dominant."},
      {"prompt": "Q: What is the difference between RAM and ROM?\nA:", "completion": "RAM (Random Access Memory) is volatile memory used for temporary data storage while a computer is running — it clears when powered off. ROM (Read-Only Memory) is non-volatile memory that stores permanent firmware or boot instructions. RAM determines how much a computer can do simultaneously; ROM holds instructions needed to start the system."},
      {"prompt": "Q: What causes inflation?\nA:", "completion": "Inflation — a general rise in price levels — is caused by several factors: demand-pull (excess consumer demand relative to supply), cost-push (rising production costs passed to consumers), and monetary expansion (too much money in circulation relative to goods). Central banks manage inflation primarily through interest rate policy."},
      {"prompt": "Q: How does the immune system recognize foreign pathogens?\nA:", "completion": "The immune system recognizes pathogens through molecular patterns called antigens — proteins on the surface of bacteria, viruses, or foreign cells. Innate immunity provides a fast, non-specific first response. The adaptive immune system then deploys B-cells that produce antibodies specific to the antigen, and T-cells that destroy infected cells. Memory cells retain the blueprint for future faster responses."},
    ],
    "hard": [
      {"prompt": "Q: A company has revenue of $5M, COGS of $2M, and operating expenses of $1.5M. What is the gross margin and operating margin?\nA:", "completion": "Gross profit = Revenue − COGS = $5M − $2M = $3M\nGross margin = $3M / $5M = 60%\n\nOperating income = Gross profit − OpEx = $3M − $1.5M = $1.5M\nOperating margin = $1.5M / $5M = 30%\n\nA 60% gross margin is strong. A 30% operating margin indicates excellent cost discipline, well above the typical 10–15% for SaaS businesses."},
      {"prompt": "Q: What is the difference between supervised, unsupervised, and reinforcement learning?\nA:", "completion": "Supervised learning trains a model on labeled input-output pairs to predict outputs for new inputs (e.g., image classification, spam detection). The signal is explicit: the correct answer is always provided.\n\nUnsupervised learning finds patterns in unlabeled data without predefined outputs (e.g., clustering customers, dimensionality reduction). The model must infer structure on its own.\n\nReinforcement learning trains an agent to take actions in an environment to maximize a cumulative reward signal. There are no labeled examples — learning comes from trial, error, and delayed feedback (e.g., game-playing agents, robotic control)."},
      {"prompt": "Q: Explain how transformer attention works and why it was a breakthrough over RNNs.\nA:", "completion": "Transformer attention allows every token in a sequence to directly attend to every other token in a single operation, computing relevance scores (queries × keys) to weight value representations. This produces context-aware embeddings regardless of token distance.\n\nRNNs process sequences step-by-step, passing a hidden state forward. This creates two problems: vanishing gradients make learning long-range dependencies hard, and sequential processing prevents parallelization.\n\nTransformers solve both: attention captures long-range dependencies in O(1) steps, and the entire sequence is processed in parallel, enabling efficient training on GPUs. This was the key breakthrough enabling large-scale language models."},
    ]
  },
  "summarization": {
    "easy": [
      {"prompt": "Summarize in one sentence: The mitochondria is a membrane-bound organelle found in the cytoplasm of eukaryotic cells. It generates most of the cell's supply of ATP, used as a source of chemical energy.", "completion": "Mitochondria are organelles in eukaryotic cells that produce ATP, the cell's primary energy currency."},
      {"prompt": "Summarize in one sentence: The Great Wall of China was built over many centuries by various Chinese dynasties to protect against invasions from the north. It stretches over 21,000 kilometers.", "completion": "The Great Wall of China is a 21,000-kilometer fortification built across centuries by Chinese dynasties to defend against northern invasions."},
      {"prompt": "Summarize in one sentence: Exercise has been shown to improve mood, increase energy levels, and reduce the risk of chronic diseases such as diabetes and heart disease.", "completion": "Regular exercise improves mood and energy while reducing the risk of chronic diseases like diabetes and heart disease."},
      {"prompt": "Summarize briefly: Blockchain is a distributed ledger technology that records transactions across many computers so that records cannot be altered retroactively. It is the underlying technology behind cryptocurrencies like Bitcoin.", "completion": "Blockchain is a tamper-resistant distributed ledger that underpins cryptocurrencies like Bitcoin by recording transactions across a decentralized network."},
    ],
    "medium": [
      {"prompt": "Summarize the following passage in 2–3 sentences:\n\nThe Amazon rainforest covers over 5.5 million square kilometers across nine countries in South America. It is home to an estimated 10% of all species on Earth and plays a critical role in regulating the global climate by absorbing vast quantities of CO2. Deforestation, driven by agriculture and logging, has destroyed about 17% of the forest in the last 50 years, threatening both biodiversity and climate stability.", "completion": "The Amazon rainforest spans 5.5 million km² across nine South American countries and harbors roughly 10% of Earth's species while acting as a major carbon sink. Deforestation for agriculture and logging has eliminated about 17% of the forest over the past five decades, posing serious risks to global biodiversity and climate stability."},
      {"prompt": "Summarize in 2–3 sentences:\n\nArtificial intelligence is increasingly being used in healthcare to assist with diagnosis, drug discovery, and treatment planning. ML models trained on large medical datasets can detect diseases like cancer from imaging data with accuracy comparable to specialists. However, concerns remain around bias in training data, explainability of model decisions, and regulatory approval processes.", "completion": "AI is transforming healthcare by enabling ML models to detect diseases such as cancer from medical imaging with specialist-level accuracy and accelerating drug discovery. Despite these advances, significant challenges remain around training data bias, the explainability of model outputs, and navigating complex regulatory frameworks."},
      {"prompt": "Summarize the key points in 2–3 sentences:\n\nRemote work has grown significantly since 2020. Studies show mixed results: some employees report higher productivity due to fewer interruptions, while others struggle with isolation, blurred work-life boundaries, and lack of career visibility. Companies are now experimenting with hybrid models to balance flexibility with collaboration.", "completion": "Remote work expanded dramatically after 2020, with productivity outcomes varying — some workers thrive with fewer interruptions while others face isolation and boundary challenges. In response, many companies are adopting hybrid models to blend the flexibility of remote work with the collaboration benefits of in-person presence."},
    ],
    "hard": [
      {"prompt": "Summarize the key arguments from both sides of this debate and provide a balanced conclusion:\n\nSome economists argue that raising the minimum wage reduces poverty and boosts consumer spending, with studies showing modest disemployment effects. Others contend it increases labor costs, forcing small businesses to cut jobs, reduce hours, or automate, disproportionately hurting low-skill workers.", "completion": "Proponents of minimum wage increases argue they directly lift incomes for low-wage workers, reduce poverty, and stimulate local economies through higher consumer spending, with empirical research showing often modest employment effects.\n\nOpponents warn that higher labor costs squeeze small business margins, potentially leading to layoffs, reduced hours, or accelerated automation — outcomes that may hurt the very workers the policy aims to help, especially in lower-cost regions.\n\nA balanced view: outcomes depend on the magnitude of the increase, local economic conditions, and implementation pace. Gradual, regionally-indexed increases appear to deliver poverty-reduction benefits while minimizing disemployment risk."},
      {"prompt": "Read the following and produce an executive summary (3–4 sentences):\n\nQuantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations. Unlike classical bits (0 or 1), qubits can exist in multiple states simultaneously. Current quantum computers from IBM, Google, and others have demonstrated 'quantum advantage' on narrow tasks but remain noisy and error-prone. Fault-tolerant quantum computing — capable of running arbitrary algorithms reliably — is estimated to be 10–20 years away. When achieved, it could break current RSA encryption and exponentially accelerate drug discovery and optimization problems.", "completion": "Quantum computing exploits superposition and entanglement to process information in ways classical computers cannot, with leading firms like IBM and Google achieving narrow quantum advantage on specific tasks. However, current systems remain noisy and error-prone, with fault-tolerant general-purpose quantum computing estimated to be 10–20 years away. When realized, the technology could compromise RSA encryption and dramatically accelerate drug discovery and complex optimization, representing both an enormous scientific opportunity and a significant cybersecurity risk."},
    ]
  },
  "chat": {
    "easy": [
      {"prompt": "<|user|>\nHi, what can you help me with?\n<|assistant|>", "completion": "Hello! I can help you with answering questions, summarizing information, following instructions, and having a conversation on a wide range of topics. What would you like to do today?"},
      {"prompt": "<|user|>\nWhat's 15% of 200?\n<|assistant|>", "completion": "15% of 200 is 30. (200 × 0.15 = 30)"},
      {"prompt": "<|user|>\nCan you recommend a book?\n<|assistant|>", "completion": "Sure! What genre do you enjoy? For example, are you into fiction, non-fiction, science, history, or self-improvement? Knowing your preferences will help me give you a great recommendation."},
      {"prompt": "<|user|>\nThank you for your help!\n<|assistant|>", "completion": "You're welcome! Feel free to come back anytime if you have more questions."},
      {"prompt": "<|user|>\nWhat time is it?\n<|assistant|>", "completion": "I don't have access to real-time data like the current time. You can check the clock on your device for the exact time!"},
    ],
    "medium": [
      {"prompt": "<|user|>\nI'm feeling overwhelmed with my workload. I have three deadlines this week.\n<|assistant|>", "completion": "That sounds stressful. Let's break it down.\n\nFirst, list all three deadlines with their due dates and rough effort required. Then prioritize by: (1) which is due soonest, and (2) which has the highest consequence if missed.\n\nOnce you have that order, time-block your day — give the highest-priority task your first 2 hours when focus is sharpest. What are the three deadlines? I can help you build a plan."},
      {"prompt": "<|user|>\nI want to learn Python. Where should I start?\n<|assistant|>\nGreat choice! Python is one of the most beginner-friendly languages. What's your goal — data science, web development, automation, or general programming?\n<|user|>\nGeneral programming, just to get started.\n<|assistant|>", "completion": "Perfect. Here's a focused starting path:\n\n1. Learn the basics: variables, data types, loops, conditionals, functions. Use the official Python tutorial (docs.python.org) or 'Automate the Boring Stuff with Python' (free online).\n2. Practice daily: solve 1–2 exercises on Exercism.io or Codewars.\n3. Build something small: a calculator, a to-do list, or a script that renames files.\n\nAim for 30 minutes a day consistently — that beats 3-hour weekend sessions. Want me to outline week 1 in more detail?"},
      {"prompt": "<|user|>\nHow do I deal with a difficult coworker who constantly interrupts me in meetings?\n<|assistant|>", "completion": "That's frustrating, and it's worth addressing directly.\n\nA few approaches depending on the relationship:\n\n1. In the moment: use a calm, neutral phrase like 'Let me finish this thought' and continue speaking without pausing.\n2. After the meeting: have a private, non-confrontational conversation. 'I've noticed I sometimes get cut off in meetings — I'd appreciate space to finish my points.'\n3. Structural fix: if you're leading the meeting, establish ground rules upfront ('Let's let each person finish before responding').\n\nMost interrupting is habitual, not malicious. A direct, low-drama approach usually resolves it."},
      {"prompt": "<|user|>\nExplain recursion to me like I'm 10.\n<|assistant|>", "completion": "Imagine you're looking for your toy in a messy room. You check each box. If a box has smaller boxes inside, you open those too — and if those have even smaller boxes, you open those as well. You keep going until you find the toy or run out of boxes to check.\n\nThat's recursion: a function that calls itself to solve a smaller version of the same problem, over and over, until it hits a simple case it can answer directly (that's called the 'base case')."},
    ],
    "hard": [
      {"prompt": "<|user|>\nI want to start a podcast but I have no audience and no equipment. Where do I start?\n<|assistant|>\nGreat goal! Let's figure out what kind of podcast you want to make first. What topic are you thinking?\n<|user|>\nI want to talk about personal finance for people in their 20s.\n<|assistant|>", "completion": "Perfect niche — personal finance for 20-somethings is evergreen and highly searchable. Here's a lean start:\n\nEquipment (under €150): Rode PodMic or Audio-Technica ATR2100x (~€80), record with Audacity (free), host on Buzzsprout or Anchor (free tier).\n\nFirst 5 episode ideas:\n1. 'The 3 money mistakes I made at 22'\n2. 'How to build your first emergency fund'\n3. 'Renting vs buying in your 20s'\n4. 'What is compound interest — and why it matters now'\n5. 'How to negotiate your first salary'\n\nAudience building: post 60-second clips on TikTok/Reels before the full episode. Consistency beats production quality early on. Want a 30-day launch timeline?"},
      {"prompt": "<|user|>\nI've been offered a job that pays 20% more but requires relocating to a new city. How do I decide?\n<|assistant|>\nThat's a meaningful decision with a lot of variables. Let's work through it together. What's driving your hesitation — is it the city itself, leaving your current network, or something else?\n<|user|>\nMainly leaving my network and my current team. I really like where I work, I just feel like the ceiling is low.\n<|assistant|>", "completion": "That's really useful clarity. The network and team attachment is real — those take years to build. But feeling like you've hit a ceiling is also real, and that tends to compound negatively over time.\n\nA few things worth weighing:\n\n1. Is the ceiling structural or situational? Could a conversation with your manager change your trajectory, or is it a company-wide constraint?\n2. Will the new city let you rebuild a network? Some cities have very active professional communities in certain fields; others are harder.\n3. Is the 20% a one-time jump or part of a higher-trajectory role? A 20% raise that comes with a faster promotion path is very different from a lateral move with more money.\n4. What does your gut say about the new company's culture and your future manager?\n\nIf the ceiling is genuinely structural at your current job, staying usually means trading a short-term comfort for long-term stagnation. The network can be rebuilt — your career momentum is harder to recover. What's the new company and role?"},
    ]
  }
}

DOMAINS = ["general", "science", "tech", "business", "creative", "medical"]
TASKS = ["instruction", "qa", "summarization", "chat"]
WEIGHTS = {"instruction": 30, "qa": 25, "summarization": 20, "chat": 25}
TOTAL_SAMPLES = 8400
STAGE_RATIOS = [0.40, 0.40, 0.20]
DIFFICULTIES = ["easy", "medium", "hard"]
NEG_PCT = 0.08

total_weight = sum(WEIGHTS.values())
records = []

for task in TASKS:
    task_share = round(WEIGHTS[task] / total_weight * TOTAL_SAMPLES)
    for di, diff in enumerate(DIFFICULTIES):
        stage_count = round(task_share * STAGE_RATIOS[di])
        pool = SAMPLES[task].get(diff) or SAMPLES[task].get("medium") or []
        for i in range(stage_count):
            base = pool[i % len(pool)]
            # Format to project-compatible Conversation format
            entry = {
                "conversations": [
                    {"role": "user", "content": base["prompt"]},
                    {"role": "assistant", "content": base["completion"]}
                ]
            }
            records.append(entry)

random.shuffle(records)

out_path = "sft_data/train.jsonl"
# Create dir if not exists
Path("sft_data").mkdir(exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Successfully integrated {len(records)} records into {out_path}")
