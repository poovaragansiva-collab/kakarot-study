# ⚡ Kakarot Study — Deployment Guide
## Launch your RAG chatbot as a public website with a custom domain

---

## 🆓 Get Your FREE Gemini API Key First

**Google Gemini 1.5 Flash is completely free** — 15 requests/min, 1500 requests/day.

1. Go to 👉 **[aistudio.google.com](https://aistudio.google.com)**
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API key"**
4. Copy the key (starts with `AIzaSy...`)
5. Keep it safe — you'll paste it into Railway/Render as a secret

That's it. No credit card. No billing. Genuinely free. ✅

---

## 🗺 Overview: Your 3-Step Launch Plan

```
Step 1: Push code to GitHub
Step 2: Deploy on Railway (free hosting, auto HTTPS)
Step 3: Connect your kakarot domain (GoDaddy / Namecheap / Hostinger)
```

Total time: ~20 minutes ⏱️

---

## STEP 1 — Push to GitHub

### 1a. Create a GitHub repository

1. Go to [github.com](https://github.com) → click **New repository**
2. Name it: `kakarot-study`
3. Set to **Public** (or Private — Railway works with both)
4. Click **Create repository**

### 1b. Push your code

```bash
cd kakarot-study/          # your project folder

git init
git add .
git commit -m "🚀 Initial commit — Kakarot Study RAG chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/kakarot-study.git
git push -u origin main
```

---

## STEP 2 — Deploy on Railway (Recommended — Free Tier)

Railway gives you: **free hosting + auto HTTPS + custom domain support**

### 2a. Sign up & connect

1. Go to [railway.app](https://railway.app) → **Sign up with GitHub**
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `kakarot-study` repo
4. Railway auto-detects the `Dockerfile` and builds it ✅

### 2b. Set your API key (SECRET — never put this in code!)

1. In Railway dashboard → click your service → **Variables** tab
2. Click **New Variable**
3. Add:
   ```
   Name:  GOOGLE_API_KEY
   Value: AIzaSy-YOUR-GEMINI-KEY-HERE
   ```
4. Railway auto-redeploys with the secret injected 🔒

### 2c. Get your Railway URL

After deploy, Railway gives you a URL like:
```
https://kakarot-study-production.up.railway.app
```
This is your app, live on the internet! Share it with anyone.

---

## STEP 3 — Connect Your Custom Domain

### 3a. Buy a "kakarot" domain

Recommended options (check availability):

| Domain | Where to buy | ~Price/year |
|--------|-------------|-------------|
| `kakarotstudy.com` | [Namecheap](https://namecheap.com) | ₹800–1,200 |
| `kakarot.study` | [Hostinger](https://hostinger.in) | ₹1,500–2,000 |
| `kakarotstudy.in` | [GoDaddy](https://godaddy.com) | ₹600–900 |
| `studykakarot.com` | [Porkbun](https://porkbun.com) | ₹700–1,000 |

> 💡 `.study` is the coolest TLD for this — `kakarot.study` looks incredibly clean.

### 3b. Add custom domain in Railway

1. Railway dashboard → your service → **Settings** → **Domains**
2. Click **Custom Domain**
3. Type your domain: `kakarot.study` (or whatever you bought)
4. Railway shows you a **CNAME record** to add, like:
   ```
   Type:  CNAME
   Name:  @  (or www)
   Value: kakarot-study-production.up.railway.app
   TTL:   Auto
   ```

### 3c. Add DNS record at your registrar

**For Namecheap:**
1. Login → Domain List → Manage → **Advanced DNS**
2. Add Record:
   - Type: `CNAME`
   - Host: `www`
   - Value: `kakarot-study-production.up.railway.app`
3. For root domain (`@`), add:
   - Type: `URL Redirect` → `https://www.kakarot.study`

**For GoDaddy:**
1. Login → My Products → DNS → Add Record
2. Same CNAME values as above

**For Hostinger:**
1. hPanel → Domains → DNS Zone
2. Add CNAME record as above

> ⏰ DNS propagation takes 5–30 minutes. After that, `kakarot.study` loads your app!

### 3d. HTTPS is automatic

Railway auto-provisions an **SSL certificate** via Let's Encrypt.
Your site will be `https://kakarot.study` — fully secure 🔒

---

## Alternative: Deploy on Render (also free)

1. Go to [render.com](https://render.com) → New → **Web Service**
2. Connect your GitHub repo
3. Render detects `render.yaml` automatically
4. Add Environment Variable: `GOOGLE_API_KEY = sk-ant-...`
5. Click **Deploy**
6. Go to **Settings → Custom Domains** → add `kakarot.study`

---

## Alternative: Streamlit Community Cloud (Easiest, but no custom domain on free plan)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub → select `kakarot-study` repo
3. Set `GOOGLE_API_KEY` in **Secrets**:
   ```toml
   GOOGLE_API_KEY = "AIzaSy-YOUR-GEMINI-KEY-HERE
   ```
4. Your app lives at: `kakarot-study.streamlit.app`
5. Custom domain requires a paid plan ($$$)

---

## 🔐 Security Checklist Before Going Public

- [ ] `GOOGLE_API_KEY` is set as an environment variable (NOT in code)
- [ ] `.gitignore` has `.env` listed (so you never accidentally push keys)
- [ ] Consider setting a usage limit in the Anthropic Console to cap costs
- [ ] Test the app yourself before sharing the URL

---

## 📁 Final File Structure

```
kakarot-study/
├── app.py                  # Streamlit UI (Kakarot branded, public-ready)
├── rag_engine.py           # RAG pipeline (extract → chunk → retrieve → generate)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config for Railway/Render
├── railway.toml            # Railway deployment config
├── render.yaml             # Render deployment config
├── .streamlit/
│   └── config.toml         # Streamlit server + theme config
├── .gitignore              # Keeps secrets out of git
└── DEPLOY.md               # This file
```

---

## 💸 Cost Estimate

| Service | Cost |
|---------|------|
| Railway hosting | Free (500 hrs/month) |
| Domain (kakarot.study) | ~₹1,500/year |
| Gemini API | ~₹0.02–0.08 per conversation (very cheap) |
| **Total** | **~₹1,500/year (domain only!)** |

---

## 🚀 Share Your App

Once live, share:
```
🎓 Try Kakarot Study — Upload your question paper and get instant AI explanations!
👉 https://kakarot.study
```

---

Built with ❤️ using LangChain, Streamlit, Google Gemini, and Railway
