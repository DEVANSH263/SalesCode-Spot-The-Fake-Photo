# Deploying the live demo

This repo already has `app.py` (Gradio webcam demo) tested and working locally.
Two things are left, and both need **your own login** (I can't do these for you):

1. Push the code to **GitHub** (for the assignment submission).
2. Push the demo to **Hugging Face Spaces** (for the free live link).

---

## 1. Push to GitHub

```powershell
cd "c:\Desktop\ML Latest Assignment"

# Log in once (opens browser)
gh auth login

# Create the repo and push in one step
gh repo create spot-the-fake-photo --public --source=. --remote=origin --push
```

That's it — `gh` creates the repo on your account and pushes the existing commit.

---

## 2. Deploy the live camera demo on Hugging Face Spaces

### One-time setup

```powershell
# Log in once (creates a token-based login, opens browser)
huggingface-cli login
```

### Create the Space and push

```powershell
cd "c:\Desktop\ML Latest Assignment\ML Latest Assignment"

# Create a new Space (sdk=gradio). Replace YOUR_USERNAME.
hf repo create spot-the-fake-photo --type space --space_sdk gradio

# Clone it next to this folder, then copy the demo files in
cd ..
git clone https://huggingface.co/spaces/YOUR_USERNAME/spot-the-fake-photo
Copy-Item "ML Latest Assignment\app.py" "spot-the-fake-photo\"
Copy-Item "ML Latest Assignment\predict.py" "spot-the-fake-photo\"
Copy-Item "ML Latest Assignment\features.py" "spot-the-fake-photo\"
Copy-Item "ML Latest Assignment\model.pkl" "spot-the-fake-photo\"
Copy-Item "ML Latest Assignment\model_nn.pt" "spot-the-fake-photo\"
Copy-Item "ML Latest Assignment\requirements.txt" "spot-the-fake-photo\"
Copy-Item "ML Latest Assignment\README_SPACE.md" "spot-the-fake-photo\README.md"

cd spot-the-fake-photo
git add .
git commit -m "Add screen-vs-real detector demo"
git push
```

Within ~1-2 minutes the Space will build and go live at:

```
https://huggingface.co/spaces/YOUR_USERNAME/spot-the-fake-photo
```

Open it, click the webcam box, allow camera access, and you'll see the
live score update every ~0.5s.

### Add the link back to your GitHub README

Once you have the live URL, add it near the top of `README.md`:

```markdown
🔴 **Live demo:** https://huggingface.co/spaces/YOUR_USERNAME/spot-the-fake-photo
```

---

## Notes

- The free Hugging Face CPU tier (2 vCPU / 16 GB RAM) is enough for this
  model — no GPU needed, no payment info required.
- The Space sleeps after a period of no traffic and wakes up again
  (~10-20s) on the next visit — totally fine for a take-home demo link.
- If `gh` or `huggingface-cli` ask for auth and a browser doesn't open
  automatically, copy the URL/code they print into any browser manually.
