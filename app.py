"""
app.py  –  Ardent LMS Chatbot API
Memory-optimised for Render free tier (512 MB).

Strategy:
  • No NLTK downloads at runtime (they consume 200+ MB)
  • No sklearn model pickle loading (RandomForest uses 150-400 MB)
  • Pure-Python keyword matching for the chatbot  (< 5 MB)
  • Lightweight TF-IDF built in-process for career recommender (< 20 MB)
  • Lazy initialisation — heavy objects built once on first request
"""

import json
import math
import os
import random
import re
import string
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
#  INTENTS  (embedded directly – no file I/O needed at runtime)
# ─────────────────────────────────────────────────────────────────────────────
INTENTS = [
    {"tag": "greeting",
     "patterns": ["hi","hello","hey","good morning","good afternoon","good evening","hey there","what's up","howdy","namaste","hii"],
     "responses": ["Hello! Welcome to Ardent Computech. How can I help you today?",
                   "Hi there! I'm your course advisor. What would you like to know?",
                   "Hey! Great to see you. Ask me anything about our courses or career guidance!"]},

    {"tag": "goodbye",
     "patterns": ["bye","goodbye","see you","take care","later","quit","exit","good night"],
     "responses": ["Goodbye! Best of luck with your learning journey!",
                   "See you soon! Feel free to come back anytime.",
                   "Bye! Keep learning and growing!"]},

    {"tag": "thanks",
     "patterns": ["thanks","thank you","thank you so much","thanks a lot","appreciate it","that's helpful","many thanks","thx"],
     "responses": ["You're welcome! Anything else I can help you with?",
                   "Happy to help! Feel free to ask more questions.",
                   "Glad I could assist! Is there anything else?"]},

    {"tag": "about_company",
     "patterns": ["tell me about ardent","what is ardent computech","about your company","who are you","what does your company do","about yourself"],
     "responses": ["Ardent Computech Private Limited is a premier IT training institute based in Kolkata. We offer 60+ courses in software development, data science, cloud computing, and more. We focus on hands-on, industry-relevant training with 100% placement support."]},

    {"tag": "contact",
     "patterns": ["contact details","phone number","email address","how to reach you","where are you located","address","contact info"],
     "responses": ["📍 Location: Salt Lake Sector V, Kolkata, West Bengal\n📞 Phone: +91 12345 78900\n📧 Email: info@ardent.ac.in\n🌐 Monday–Saturday, 9AM–7PM"]},

    {"tag": "placement",
     "patterns": ["placement assistance","job placement","help with jobs","placement rate","placement support","get job after course","placement guarantee","100 percent placement","job after training"],
     "responses": ["We provide 100% placement assistance!\n✅ Resume building\n✅ Mock interviews\n✅ Industry connections\n✅ Job portal access\n✅ Soft skills training\n\nOver 500+ students placed in top MNCs!"]},

    {"tag": "fees",
     "patterns": ["course fees","how much does it cost","what is the price","fee structure","cost of course","is it affordable","pricing","emi option","fee details","how much to pay","fees"],
     "responses": ["Course fees vary by program:\n💻 Short courses: ₹5,000–₹15,000\n📚 Professional courses: ₹15,000–₹35,000\n🎓 Industrial training: ₹8,000–₹20,000\n\nEMI options available! Contact us for exact pricing."]},

    {"tag": "duration",
     "patterns": ["how long is the course","course duration","how many months","time to complete","course length","how many weeks","course timing"],
     "responses": ["Course durations:\n⏱ 30-day intensive: Short skills\n📅 2–3 months: Most professional courses\n🗓 6 months: Full-stack / Data Science\n📆 1 year: Advanced programs\n\nWeekend batches also available!"]},

    {"tag": "java",
     "patterns": ["java course","learn java","java programming","java training","core java","advanced java","java developer","java fees","java duration","java certification","java syllabus"],
     "responses": ["☕ Java Development Course:\n\n📚 Core Java → Advanced Java → Spring Boot → Hibernate → REST API\n⏱ Duration: 3–4 months\n💰 Fee: ₹18,000–₹22,000\n🏆 Certification included\n💼 Placement support: YES\n\nIdeal for: Beginners and professionals moving to Java backend."]},

    {"tag": "python",
     "patterns": ["python course","learn python","python programming","python training","python developer","python scripting","python fees","python syllabus","python basics"],
     "responses": ["🐍 Python Programming Course:\n\n📚 Fundamentals → OOP → NumPy/Pandas → Flask/Django → Web Scraping\n⏱ Duration: 2–3 months\n💰 Fee: ₹12,000–₹18,000\n🏆 Certificate provided\n\nBest for: Students, beginners, data science aspirants."]},

    {"tag": "php",
     "patterns": ["php course","learn php","php training","php developer","php web development","php laravel","php mysql","php syllabus"],
     "responses": ["🌐 PHP Web Development Course:\n\n📚 PHP basics → MySQL → OOP → Laravel → REST APIs → MVC\n⏱ Duration: 3 months\n💰 Fee: ₹14,000–₹18,000\n💼 Live project included\n\nIdeal for: Web dev beginners, freelancers, backend devs."]},

    {"tag": "mern",
     "patterns": ["mern stack","mern course","mongodb express react node","learn mern","full stack mern","mern training","mern developer","mern syllabus","mern fees"],
     "responses": ["⚡ MERN Stack Course:\n\n📚 MongoDB → Express.js → React.js → Node.js → REST API → JWT → Redux\n⏱ Duration: 4–5 months\n💰 Fee: ₹25,000–₹30,000\n💼 Full-stack job ready!\n\nBest for: Full-stack web dev career."]},

    {"tag": "mean",
     "patterns": ["mean stack","mean course","angular node express mongodb","learn mean","mean training","mean developer","mean syllabus"],
     "responses": ["🔷 MEAN Stack Course:\n\n📚 MongoDB → Express.js → Angular (TypeScript) → Node.js → RxJS → NgRx\n⏱ Duration: 4–5 months\n💰 Fee: ₹25,000–₹30,000\n\nBest for: Enterprise Angular apps."]},

    {"tag": "spring_boot",
     "patterns": ["spring boot","spring course","spring framework","spring boot training","spring mvc","spring boot rest api","spring boot microservices","java spring"],
     "responses": ["🌱 Spring Boot & Microservices Course:\n\n📚 Spring Core → Spring MVC → Spring Boot → JPA → Security → Microservices → Docker\n⏱ Duration: 3 months\n💰 Fee: ₹20,000–₹25,000\n\nPrerequisite: Core Java."]},

    {"tag": "data_science",
     "patterns": ["data science course","learn data science","data science training","data analyst","become data scientist","data science syllabus","data science fees","data science career"],
     "responses": ["📊 Data Science Course:\n\n📚 Python → Statistics → NumPy/Pandas → ML Algorithms → Scikit-learn → Visualization → Capstone project\n⏱ Duration: 5–6 months\n💰 Fee: ₹28,000–₹35,000\n\nIdeal for: Math/stats graduates, engineers."]},

    {"tag": "machine_learning",
     "patterns": ["machine learning course","ml course","learn ml","machine learning training","ml algorithms","supervised learning","deep learning","neural networks","ml engineer","ai ml course","artificial intelligence machine learning"],
     "responses": ["🤖 Machine Learning & AI Course:\n\n📚 ML fundamentals → Supervised/Unsupervised → Neural Networks → TensorFlow → NLP basics → Model deployment\n⏱ Duration: 5–6 months\n💰 Fee: ₹30,000–₹38,000\n\nPrerequisite: Python + basic statistics."]},

    {"tag": "flutter",
     "patterns": ["flutter course","flutter training","learn flutter","flutter developer","flutter dart","cross platform app","flutter mobile","flutter fees","flutter syllabus"],
     "responses": ["📱 Flutter App Development:\n\n📚 Dart → Widgets → State Management → REST API → Firebase → App Store deployment\n⏱ Duration: 3–4 months\n💰 Fee: ₹20,000–₹25,000\n\nIdeal for: Mobile developers."]},

    {"tag": "react",
     "patterns": ["react course","react js","learn react","react training","react developer","react native","react syllabus","react hooks","react redux"],
     "responses": ["⚛️ React.js Course:\n\n📚 JS ES6+ → Components → Hooks → Redux → React Router → Next.js basics\n⏱ Duration: 2–3 months\n💰 Fee: ₹15,000–₹20,000\n\nIdeal for: Frontend developers."]},

    {"tag": "angular",
     "patterns": ["angular course","learn angular","angular training","angular developer","angular typescript","angular material","angular fees"],
     "responses": ["🔵 Angular Course:\n\n📚 TypeScript → Components → Services → Routing → Reactive Forms → HTTP → Angular Material\n⏱ Duration: 2–3 months\n💰 Fee: ₹15,000–₹18,000"]},

    {"tag": "nodejs",
     "patterns": ["node js course","nodejs training","learn node","node developer","express js","backend javascript","nodejs syllabus","api development node"],
     "responses": ["🟢 Node.js Backend:\n\n📚 Node fundamentals → Express → REST API → MongoDB → JWT → WebSockets → Deployment\n⏱ Duration: 2–3 months\n💰 Fee: ₹14,000–₹18,000"]},

    {"tag": "data_analytics",
     "patterns": ["data analytics course","business analytics","data analyst training","excel power bi","tableau course","analytics tools","data visualization course","analyst career"],
     "responses": ["📈 Data Analytics Course:\n\n📚 Advanced Excel → SQL → Python/Pandas → Power BI → Tableau → Data Storytelling\n⏱ Duration: 3–4 months\n💰 Fee: ₹18,000–₹24,000\n\nIdeal for: MBA students, business professionals."]},

    {"tag": "cloud",
     "patterns": ["cloud computing course","aws course","azure training","google cloud","cloud certification","devops cloud","cloud engineer","cloud training","cloud fees"],
     "responses": ["☁️ Cloud Computing Course:\n\n📚 AWS core services → Azure → Docker → Kubernetes → CI/CD → Cloud security\n⏱ Duration: 3–4 months\n💰 Fee: ₹22,000–₹28,000\n🏆 AWS/Azure exam prep included"]},

    {"tag": "devops",
     "patterns": ["devops course","learn devops","devops training","ci cd pipeline","docker kubernetes","devops engineer","devops syllabus","devops fees","jenkins git devops"],
     "responses": ["⚙️ DevOps Engineering Course:\n\n📚 Linux → Git → Docker → Kubernetes → Jenkins → Ansible → Terraform → Monitoring\n⏱ Duration: 3–4 months\n💰 Fee: ₹22,000–₹28,000"]},

    {"tag": "cybersecurity",
     "patterns": ["cybersecurity course","ethical hacking","network security","ceh course","security training","cyber security fees","penetration testing","bug bounty"],
     "responses": ["🔐 Cybersecurity & Ethical Hacking:\n\n📚 Networking → Linux → Web App Security → Penetration Testing → Kali Linux → OWASP → CEH prep\n⏱ Duration: 3–4 months\n💰 Fee: ₹20,000–₹25,000"]},

    {"tag": "ui_ux",
     "patterns": ["ui ux course","user interface design","user experience","figma course","web design course","ui design training","ux designer","design course fees"],
     "responses": ["🎨 UI/UX Design Course:\n\n📚 Design principles → Wireframing → Figma (complete) → Adobe XD → Usability testing → Portfolio\n⏱ Duration: 2–3 months\n💰 Fee: ₹12,000–₹18,000"]},

    {"tag": "android",
     "patterns": ["android course","android development","learn android","kotlin course","java android","android app development","android training","android fees"],
     "responses": ["📲 Android App Development:\n\n📚 Kotlin → Android Studio → Fragments → SQLite/Room → Retrofit → Firebase → Play Store\n⏱ Duration: 3–4 months\n💰 Fee: ₹18,000–₹24,000"]},

    {"tag": "sql",
     "patterns": ["sql course","learn sql","database course","mysql training","sql developer","database management","sql queries","sql syllabus","dbms course"],
     "responses": ["🗄️ SQL & Database Management:\n\n📚 SQL fundamentals → Joins → Stored Procedures → MySQL → PostgreSQL → Indexing\n⏱ Duration: 1.5–2 months\n💰 Fee: ₹8,000–₹12,000"]},

    {"tag": "blockchain",
     "patterns": ["blockchain course","learn blockchain","ethereum course","smart contracts","solidity","web3 course","crypto development","blockchain training"],
     "responses": ["⛓️ Blockchain Development:\n\n📚 Blockchain fundamentals → Ethereum → Solidity → Web3.js → DeFi → NFT development\n⏱ Duration: 3 months\n💰 Fee: ₹22,000–₹28,000"]},

    {"tag": "autocad",
     "patterns": ["autocad course","cad training","mechanical design","autocad certification","2d 3d design","solidworks","catia","autocad fees"],
     "responses": ["📐 AutoCAD & Mechanical Design:\n\n📚 AutoCAD 2D → AutoCAD 3D → SolidWorks → CATIA intro → CNC basics\n⏱ Duration: 2–3 months\n💰 Fee: ₹10,000–₹16,000"]},

    {"tag": "digital_marketing",
     "patterns": ["digital marketing course","seo course","social media marketing","google ads","content marketing","marketing training","online marketing","digital marketing fees"],
     "responses": ["📣 Digital Marketing Course:\n\n📚 SEO → Google Ads → Social Media → Email Marketing → Analytics → Facebook Ads\n⏱ Duration: 2–3 months\n💰 Fee: ₹10,000–₹15,000"]},

    {"tag": "power_bi",
     "patterns": ["power bi course","learn power bi","business intelligence","power bi training","bi tools","data reporting","power bi fees"],
     "responses": ["📊 Power BI Course:\n\n📚 Power BI Desktop → Data modeling → DAX → Power Query → Dashboards → Power BI Service\n⏱ Duration: 1.5–2 months\n💰 Fee: ₹8,000–₹12,000"]},

    {"tag": "course_list",
     "patterns": ["what courses do you offer","list of courses","all courses","show me courses","course catalog","available courses","what can i learn here","all available programs"],
     "responses": ["🎓 Ardent Computech Course Catalog:\n\n💻 Programming: Java, Python, PHP, C/C++, Go, Rust\n🌐 Web Dev: MERN, MEAN, React, Angular, Node.js, Django, Laravel\n📱 Mobile: Flutter, Android, iOS\n📊 Data: Data Science, ML/AI, Data Analytics, NLP, Power BI\n☁️ Cloud: AWS, Azure, GCP, DevOps, Kubernetes\n🔐 Security: Cybersecurity, Ethical Hacking\n🗄️ Database: SQL, MongoDB, PostgreSQL\n🎨 Design: UI/UX, Figma, AutoCAD\n📣 Others: Digital Marketing, Excel, Networking, Blockchain, IoT\n\n60+ courses available!"]},

    {"tag": "batch",
     "patterns": ["batch timing","class schedule","morning batch","evening batch","weekend batch","online or offline","mode of training","online class","classroom training"],
     "responses": ["📅 Batch Options:\n\n☀️ Morning: 9:00 AM–11:00 AM\n🌆 Evening: 6:00 PM–8:00 PM\n📅 Weekend: Saturday & Sunday\n💻 Online: Live classes + recorded sessions\n🏫 Offline: Kolkata classroom\n\nFlexible timing for working professionals!"]},

    {"tag": "certificate",
     "patterns": ["do you give certificate","course certificate","certification program","industry recognized certificate","certificate after course","completion certificate","is certificate valid"],
     "responses": ["🏆 Yes! Ardent provides:\n✅ Course completion certificate\n✅ Industry-recognized certification\n✅ Digital & physical copies\n✅ LinkedIn shareable badges\n✅ Vendor cert prep (AWS, Azure, Google, Cisco)\n\nRecognized by 200+ hiring companies!"]},

    {"tag": "demo_class",
     "patterns": ["free demo class","trial class","can i try before joining","demo session","free trial","sample class","free webinar","attend demo"],
     "responses": ["🎯 Yes! We offer FREE demo classes!\n\n✅ No registration fee\n✅ Experience teaching style\n✅ Meet the instructor\n✅ Get course roadmap\n✅ Q&A session\n\nCall +91 12345 78900 or visit our website to book your FREE demo!"]},

    {"tag": "internship",
     "patterns": ["internship program","paid internship","industrial training","6 month internship","internship certificate","college internship","summer internship"],
     "responses": ["🏢 Industrial Training & Internship:\n\n📋 6-month Industrial Training (B.Tech/BCA students)\n💰 Stipend-based internships available\n🏆 Training + Internship certificate\n🔗 Live project exposure\n💼 Direct placement after training\n\nContact us for current openings!"]},

    {"tag": "career_change",
     "patterns": ["i want to change career","switch to it","non it to it","career switch","change my field","start it career","new career in technology","career transition","move to software"],
     "responses": ["🔄 Career Change to IT? Great decision!\n\nRecommended path:\n1️⃣ Python (most beginner-friendly)\n2️⃣ SQL & Data basics\n3️⃣ Choose: Data Analytics, Web Dev, or Cloud\n\nBook a FREE career consultation: +91 12345 78900\n\nMany students successfully switched from banking, teaching, and sales to software careers!"]},

    {"tag": "fresher",
     "patterns": ["i am a fresher","just graduated","no experience","college student","final year student","beginner course","starting from scratch","no coding knowledge","first time learning programming"],
     "responses": ["🌟 Welcome, fresher!\n\n🔰 Step 1: Python/Java basics\n🔰 Step 2: SQL\n🔰 Step 3: Choose your path:\n  • Web → MERN/MEAN\n  • Data → Python + ML\n  • Mobile → Flutter/Android\n\n💡 Courses designed for zero-experience beginners!\nFREE career counseling available."]},

    {"tag": "salary",
     "patterns": ["what is salary after course","expected salary","how much can i earn","it salary","package after training","software developer salary","data scientist salary","starting salary it"],
     "responses": ["💰 Expected Salary After Ardent Training:\n\n🖥️ Web Developer (Fresher): ₹3–5 LPA\n📊 Data Analyst: ₹4–7 LPA\n🤖 ML Engineer: ₹6–12 LPA\n☁️ Cloud Engineer: ₹5–10 LPA\n📱 Mobile Developer: ₹4–8 LPA\n\nTop students placed at ₹8–15 LPA!"]},

    {"tag": "projects",
     "patterns": ["do you provide projects","live project","real projects","portfolio project","hands on project","capstone project","industry project","project work in course"],
     "responses": ["🚀 Yes! All courses include real-world projects!\n\n✅ 2–5 projects per course\n✅ Industry-relevant problems\n✅ GitHub portfolio setup\n✅ Project presentation support\n✅ Some courses include live client projects"]},

    {"tag": "scholarship",
     "patterns": ["scholarship available","discount on fees","financial help","concession","free course","subsidized fees","fee waiver","scholarship for students"],
     "responses": ["🎓 Fee Concession Programs:\n\n✅ Merit scholarship: Up to 30% off\n✅ Early bird discount: 15% off\n✅ Referral discount: ₹2,000 off per referral\n✅ Group discount: 20% off for 3+\n✅ SC/ST special concession\n\nContact: +91 12345 78900"]},

    {"tag": "not_understood",
     "patterns": [],
     "responses": ["I'm not sure I understood that. You can ask about:\n• Specific courses (Java, Python, Data Science, etc.)\n• Course fees and duration\n• Career guidance\n• Placement support\n• Batch timings\n\nOr type 'courses' to see all programs!"]},
]

# ─────────────────────────────────────────────────────────────────────────────
#  CAREER GUIDANCE DATA  (embedded – no CSV file I/O)
# ─────────────────────────────────────────────────────────────────────────────
CAREER_DATA = [
    {"background":"non-technical","field":"Commerce","current_role":"Student","experience_years":0,"skills":"Excel basic","interest":"data analytics","goal":"corporate job","education":"B.Com","primary_course":"Advanced Excel","related_courses":"Advanced Excel,SQL,Power BI,Data Analytics","reason":"Commerce background with data skills = business analyst role; Excel and SQL are must-have foundations."},
    {"background":"technical","field":"Computer Science","current_role":"Student","experience_years":0,"skills":"none","interest":"web development","goal":"get job","education":"B.Tech","primary_course":"Python","related_courses":"Python,HTML/CSS,MERN Stack","reason":"Python is the best first language; MERN gives full-stack web skills for high-demand jobs."},
    {"background":"technical","field":"Computer Science","current_role":"Student","experience_years":0,"skills":"none","interest":"data science","goal":"get job","education":"B.Tech","primary_course":"Python","related_courses":"Python,Data Science,SQL","reason":"Data science is a high-paying field; Python and SQL are essential foundations."},
    {"background":"technical","field":"Information Technology","current_role":"Student","experience_years":0,"skills":"Java basics","interest":"software development","goal":"get job","education":"BCA","primary_course":"Core Java","related_courses":"Core Java,Spring Boot,SQL","reason":"Java is enterprise-standard; Spring Boot for backend development."},
    {"background":"technical","field":"Computer Science","current_role":"Python Developer","experience_years":2,"skills":"Python Django","interest":"data science","goal":"AI career","education":"B.Tech","primary_course":"Machine Learning","related_courses":"Machine Learning,Deep Learning,NLP","reason":"Python devs transitioning to AI/ML - natural progression with highest pay."},
    {"background":"technical","field":"IT","current_role":"System Administrator","experience_years":5,"skills":"Linux networking","interest":"cloud","goal":"cloud engineer","education":"B.Tech","primary_course":"AWS","related_courses":"AWS,Azure,DevOps,Kubernetes","reason":"Sysadmins moving to cloud - best career upgrade in IT."},
    {"background":"non-technical","field":"Banking","current_role":"Bank Officer","experience_years":5,"skills":"none","interest":"IT transition","goal":"career change","education":"B.Com","primary_course":"Python","related_courses":"Python,Data Science,SQL","reason":"Python and data science are best for banking sector tech transition."},
    {"background":"non-technical","field":"Arts","current_role":"Student","experience_years":0,"skills":"MS Office","interest":"digital marketing","goal":"freelance","education":"BA","primary_course":"Digital Marketing","related_courses":"Digital Marketing,Advanced Excel,UI/UX","reason":"Creative background suits digital marketing; Excel for analytics."},
    {"background":"technical","field":"Computer Science","current_role":"Software Developer","experience_years":2,"skills":"JavaScript React","interest":"full stack","goal":"salary hike","education":"B.Tech","primary_course":"Node.js","related_courses":"Node.js,MongoDB,Docker,AWS","reason":"Adding backend and DevOps to React skills = full-stack premium salary."},
    {"background":"non-technical","field":"Management","current_role":"Business Analyst","experience_years":4,"skills":"Excel presentations","interest":"data analytics","goal":"better salary","education":"MBA","primary_course":"Power BI","related_courses":"Advanced Excel,SQL,Power BI,Data Analytics","reason":"MBA + data skills = high-value business analyst role."},
    {"background":"technical","field":"Mechanical","current_role":"Mechanical Engineer","experience_years":3,"skills":"AutoCAD basic","interest":"design software","goal":"upskill","education":"B.Tech Mech","primary_course":"AutoCAD","related_courses":"AutoCAD Advanced,SolidWorks,CATIA","reason":"Mechanical engineers need advanced CAD and 3D modeling for better roles."},
    {"background":"technical","field":"Computer Science","current_role":"Frontend Developer","experience_years":2,"skills":"React HTML CSS","interest":"backend","goal":"full stack","education":"B.Tech","primary_course":"Node.js","related_courses":"Node.js,Express.js,MongoDB,REST API","reason":"Frontend devs adding backend = full-stack premium salary."},
    {"background":"non-technical","field":"Sales","current_role":"Sales Executive","experience_years":5,"skills":"communication","interest":"digital marketing","goal":"online business","education":"Any","primary_course":"Digital Marketing","related_courses":"Digital Marketing,Social Media,Google Ads","reason":"Sales skills + digital marketing = high-value modern marketer."},
    {"background":"technical","field":"Computer Science","current_role":"Mobile Developer","experience_years":3,"skills":"Android Java","interest":"cross platform","goal":"new skills","education":"B.Tech","primary_course":"Flutter","related_courses":"Flutter,React Native,Dart","reason":"Flutter is the future of cross-platform; adds iOS capability."},
    {"background":"non-technical","field":"HR","current_role":"HR Professional","experience_years":6,"skills":"recruitment tools","interest":"analytics","goal":"data-driven HR","education":"MBA HR","primary_course":"Advanced Excel","related_courses":"Advanced Excel,SQL,Power BI,Data Analytics","reason":"HR analytics is the future; data skills multiply HR value."},
    {"background":"technical","field":"IT","current_role":"Network Engineer","experience_years":4,"skills":"CCNA networking","interest":"cloud networking","goal":"cloud career","education":"B.Tech","primary_course":"AWS","related_courses":"AWS Networking,Azure,DevOps,Kubernetes","reason":"Network engineers in demand for cloud infrastructure."},
    {"background":"technical","field":"Computer Science","current_role":"Data Analyst","experience_years":3,"skills":"SQL Excel Power BI","interest":"machine learning","goal":"data scientist","education":"B.Tech","primary_course":"Python","related_courses":"Python,Machine Learning,Statistical Analysis,MLflow","reason":"Data analysts upgrading to data scientist with ML skills."},
    {"background":"non-technical","field":"Teaching","current_role":"School Teacher","experience_years":7,"skills":"MS Office","interest":"online teaching","goal":"upskill","education":"B.Ed","primary_course":"Digital Marketing","related_courses":"Advanced Excel,Digital Marketing,UI/UX","reason":"Teachers can expand to ed-tech and content creation."},
    {"background":"technical","field":"Computer Science","current_role":"Student","experience_years":0,"skills":"C C++","interest":"competitive programming","goal":"top MNC","education":"B.Tech","primary_course":"Core Java","related_courses":"Data Structures,Java,Python,MERN Stack","reason":"DSA + web skills = MNC placement package."},
    {"background":"non-technical","field":"Finance","current_role":"Financial Analyst","experience_years":5,"skills":"Excel","interest":"financial modeling","goal":"fintech","education":"MBA Finance","primary_course":"Python","related_courses":"Python,SQL,Power BI,Data Analytics","reason":"Fintech demands financial analysts with Python and data skills."},
    {"background":"technical","field":"Computer Science","current_role":"Student","experience_years":0,"skills":"none","interest":"cybersecurity","goal":"ethical hacking career","education":"B.Tech","primary_course":"Cybersecurity","related_courses":"Networking,Cybersecurity,Ethical Hacking,Linux","reason":"Cybersecurity is booming; structured path from networking to hacking."},
    {"background":"technical","field":"Electronics","current_role":"Student","experience_years":0,"skills":"embedded C","interest":"hardware programming","goal":"get job","education":"B.Tech ECE","primary_course":"Embedded Systems","related_courses":"Embedded Systems,IoT,Python","reason":"ECE students can combine hardware and software for IoT careers."},
    {"background":"non-technical","field":"Journalism","current_role":"Content Writer","experience_years":3,"skills":"writing SEO","interest":"digital marketing","goal":"freelance","education":"BA Journalism","primary_course":"Digital Marketing","related_courses":"Digital Marketing,SEO,Content Strategy","reason":"Writers become highly-paid digital marketers with tech skills."},
    {"background":"technical","field":"IT","current_role":"Cloud Engineer","experience_years":4,"skills":"AWS basics","interest":"advanced cloud","goal":"AWS certification","education":"B.Tech","primary_course":"AWS","related_courses":"AWS Advanced,Terraform,Kubernetes,DevOps","reason":"Advanced AWS + IaC tools = senior cloud architect salary."},
    {"background":"technical","field":"Computer Science","current_role":"Student","experience_years":0,"skills":"HTML basics","interest":"web design","goal":"frontend job","education":"B.Sc CS","primary_course":"React.js","related_courses":"React.js,JavaScript,CSS Advanced,Figma","reason":"Modern frontend needs React + design skills for good package."},
    {"background":"non-technical","field":"Pharmacy","current_role":"Pharmacist","experience_years":6,"skills":"none","interest":"health data","goal":"pharma analytics","education":"B.Pharm","primary_course":"Data Analytics","related_courses":"Python,Data Science,SQL","reason":"Pharmacists with data skills = clinical data analyst career."},
    {"background":"technical","field":"Computer Science","current_role":"Student","experience_years":0,"skills":"Java basics","interest":"Android app","goal":"mobile developer","education":"BCA","primary_course":"Android Development","related_courses":"Android Kotlin,Firebase,REST API,Material Design","reason":"Solid Android path for BCA students seeking mobile dev career."},
    {"background":"non-technical","field":"Entrepreneur","current_role":"Business Owner","experience_years":0,"skills":"basic office","interest":"grow business online","goal":"digital business","education":"MBA","primary_course":"Digital Marketing","related_courses":"Digital Marketing,E-commerce,Data Analytics,Excel","reason":"Business owners need digital marketing + analytics to grow."},
    {"background":"technical","field":"IT","current_role":"DevOps Engineer","experience_years":4,"skills":"Jenkins Docker","interest":"Kubernetes advanced","goal":"senior DevOps","education":"B.Tech","primary_course":"Kubernetes","related_courses":"Kubernetes Advanced,Terraform,AWS,GitOps","reason":"Senior DevOps needs IaC and advanced orchestration."},
    {"background":"non-technical","field":"Interior Design","current_role":"Interior Designer","experience_years":7,"skills":"AutoCAD 2D Sketchup","interest":"3D visualization","goal":"advanced design","education":"B.Des","primary_course":"AutoCAD","related_courses":"AutoCAD 3D,3ds Max,Lumion,BIM Revit","reason":"Interior designers need photorealistic 3D rendering tools."},
]

# ─────────────────────────────────────────────────────────────────────────────
#  LIGHTWEIGHT PURE-PYTHON NLP UTILS
#  No NLTK, no scikit-learn → tiny memory footprint
# ─────────────────────────────────────────────────────────────────────────────

_STOP = {
    "a","an","the","is","it","in","of","to","and","or","for","on","at",
    "by","with","this","that","are","was","be","as","do","does","can",
    "i","me","my","we","you","your","he","she","they","our","will","would",
    "should","could","have","has","had","not","no","so","if","but","then",
    "than","when","where","which","who","how","what","why","please","just",
    "want","need","like","help","tell","know","get","give","make","take",
    "any","all","some","more","about","also","am","its","their","there",
    "from","into","up","out","been","being","very","too","much","many",
    "over","own","same","other","well","s","t","re","ve","ll","d",
}

def _tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w and w not in _STOP and len(w) > 1]

def _tfidf_vectors(corpus):
    """
    Compute TF-IDF for a list of token-lists.
    Returns (vocab: {term->idx}, matrix: list of {term->tfidf}).
    """
    N = len(corpus)
    df = defaultdict(int)
    for doc in corpus:
        for term in set(doc):
            df[term] += 1

    vocab = {t: i for i, t in enumerate(df)}

    vecs = []
    for doc in corpus:
        tf = defaultdict(int)
        for t in doc:
            tf[t] += 1
        total = len(doc) or 1
        vec = {}
        for t, cnt in tf.items():
            idf = math.log((N + 1) / (df[t] + 1)) + 1
            vec[t] = (cnt / total) * idf
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1
        vecs.append({t: v / norm for t, v in vec.items()})
    return vocab, vecs

def _cosine(a: dict, b: dict) -> float:
    return sum(a.get(t, 0) * v for t, v in b.items())


# ─────────────────────────────────────────────────────────────────────────────
#  CHATBOT  –  pattern keyword index  (O(1) lookup)
# ─────────────────────────────────────────────────────────────────────────────

# Build: keyword → list of (tag, weight)
_KEYWORD_INDEX: dict[str, list] = defaultdict(list)

def _build_chatbot_index():
    for intent in INTENTS:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            tokens = _tokenize(pattern)
            for tok in tokens:
                _KEYWORD_INDEX[tok].append(tag)

_build_chatbot_index()

_TAG_RESPONSES = {intent["tag"]: intent["responses"] for intent in INTENTS}


def _predict_intent(text: str, threshold: float = 0.15) -> str:
    tokens = _tokenize(text)
    if not tokens:
        return "not_understood"

    scores: dict[str, float] = defaultdict(float)
    for tok in tokens:
        for tag in _KEYWORD_INDEX.get(tok, []):
            scores[tag] += 1.0 / len(tokens)   # normalise by query length

    # Exact multi-gram boost
    bigrams = [tokens[i] + " " + tokens[i+1] for i in range(len(tokens) - 1)]
    for bg in bigrams:
        for tag in _KEYWORD_INDEX.get(bg, []):
            scores[tag] += 1.5 / len(tokens)

    if not scores:
        return "not_understood"

    best_tag = max(scores, key=scores.__getitem__)
    if scores[best_tag] < threshold:
        return "not_understood"
    return best_tag


# ─────────────────────────────────────────────────────────────────────────────
#  CAREER RECOMMENDER  –  TF-IDF cosine similarity
#  Built lazily on first /recommend request
# ─────────────────────────────────────────────────────────────────────────────

_career_vecs  = None   # list of (tfidf_dict, record)
_career_ready = False

def _build_career_index():
    global _career_vecs, _career_ready
    corpus_tokens = []
    for rec in CAREER_DATA:
        parts = []
        for k in ("background","field","current_role","skills","interest","goal","education"):
            v = str(rec.get(k,"")).strip()
            if v and v.lower() not in ("none","nan","0",""):
                parts.append(f"{k} {v}")
        corpus_tokens.append(_tokenize(" ".join(parts)))

    _, vecs = _tfidf_vectors(corpus_tokens)
    _career_vecs  = list(zip(vecs, CAREER_DATA))
    _career_ready = True


def _recommend(profile: dict, top_k: int = 3):
    global _career_ready
    if not _career_ready:
        _build_career_index()

    parts = []
    for k in ("background","field","current_role","experience_years","skills","interest","goal","education"):
        v = str(profile.get(k,"")).strip()
        if v and v.lower() not in ("none","nan","0",""):
            parts.append(f"{k} {v}")

    if not parts:
        return []

    q_tokens = _tokenize(" ".join(parts))
    if not q_tokens:
        return []

    # Quick TF for query
    tf = defaultdict(int)
    for t in q_tokens:
        tf[t] += 1
    total = len(q_tokens)
    q_vec = {t: cnt/total for t, cnt in tf.items()}
    norm  = math.sqrt(sum(v*v for v in q_vec.values())) or 1
    q_vec = {t: v/norm for t, v in q_vec.items()}

    scored = []
    for vec, rec in _career_vecs:
        sim = _cosine(q_vec, vec)
        scored.append((sim, rec))

    scored.sort(key=lambda x: -x[0])

    # Deduplicate by primary_course, take top_k
    seen, results = set(), []
    for sim, rec in scored:
        pc = rec["primary_course"]
        if pc not in seen:
            seen.add(pc)
            results.append({
                "primary_course":  pc,
                "related_courses": rec["related_courses"],
                "reason":          rec["reason"],
                "confidence":      round(float(sim), 3),
            })
        if len(results) >= top_k:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":    "Ardent Chatbot API is running",
        "version":   "2.0-lite",
        "endpoints": {
            "chat":      "POST /chat      → { message }",
            "recommend": "POST /recommend → { background, field, current_role, experience_years, skills, interest, goal, education }",
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "message field required"}), 400

    user_msg = str(data["message"]).strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    tag      = _predict_intent(user_msg)
    response = random.choice(_TAG_RESPONSES.get(tag, _TAG_RESPONSES["not_understood"]))

    return jsonify({
        "response":   response,
        "tag":        tag,
        "confidence": 1.0 if tag != "not_understood" else 0.0,
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    recs = _recommend(data)
    if not recs:
        return jsonify({"error": "At least one profile field required"}), 400

    return jsonify({
        "recommendations": recs,
        "profile_summary": " | ".join(
            f"{k}: {data[k]}" for k in
            ("background","field","current_role","skills","interest","goal")
            if data.get(k)
        ),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
