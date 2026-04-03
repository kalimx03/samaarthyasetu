"""
SamaarthyaSetu PyTorch Skill Matching Model
Learns to score candidate-job compatibility using embeddings.
Uses skill overlap, accommodation coverage, and disability compatibility signals.
"""
from __future__ import annotations
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

# ─── Feature Engineering ──────────────────────────────────────────────────────

ALL_SKILLS = [
    "screen reader", "data entry", "ms excel", "python basics", "tailoring",
    "embroidery", "fabric cutting", "design basics", "coding", "javascript",
    "react", "git", "bakery", "packaging", "sorting", "customer greeting",
    "accounting basics", "tally", "ms office", "content writing", "social media",
    "blogging", "retail operations", "pos system", "inventory", "customer service",
    "quality check", "pattern recognition", "excel", "handicrafts", "pottery", "art",
    "python", "data analysis", "typing", "bpo", "customer support", "crm tools",
    "english communication", "textile weaving", "handloom", "color design",
    "pattern making", "food packaging", "cleaning", "stock sorting",
    "content moderation", "data tagging", "candle making", "soap making",
    "soft skills", "data entry", "it skills",
]

ALL_DISABILITIES = [
    "visual impairment", "hearing impairment", "locomotor disability",
    "intellectual disability", "speech & language disability", "low vision",
    "autism spectrum", "cerebral palsy", "muscular dystrophy", "deaf-blind",
]

ALL_ACCOMMODATIONS = [
    "screen reader software", "large font displays", "sign language interpreter",
    "visual alerts", "wheelchair access", "ergonomic workstation", "text-based communication",
    "chat tools", "text-to-speech", "quiet workspace", "structured routines", "no open office",
    "visual cues", "text messaging only", "large monitors", "high contrast ui",
    "magnification software", "work from home option", "ergonomic chair",
    "adapted loom equipment", "seated work only", "job coach", "visual task boards",
    "buddy system", "full remote", "voice-to-text software", "no physical strain tasks",
    "braille materials", "tactile sign language interpreter",
]


def skill_to_vector(skills: list[str]) -> torch.Tensor:
    vec = torch.zeros(len(ALL_SKILLS))
    for skill in skills:
        sk = skill.lower().strip()
        if sk in ALL_SKILLS:
            vec[ALL_SKILLS.index(sk)] = 1.0
    return vec


def accommodation_to_vector(needs: list[str]) -> torch.Tensor:
    vec = torch.zeros(len(ALL_ACCOMMODATIONS))
    for need in needs:
        n = need.lower().strip()
        if n in ALL_ACCOMMODATIONS:
            vec[ALL_ACCOMMODATIONS.index(n)] = 1.0
    return vec


def disability_to_vector(disability_type: str) -> torch.Tensor:
    vec = torch.zeros(len(ALL_DISABILITIES))
    dt = disability_type.lower().strip()
    if dt in ALL_DISABILITIES:
        vec[ALL_DISABILITIES.index(dt)] = 1.0
    return vec


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CandidateJobDataset(Dataset):
    """
    Dataset of (candidate_features, job_features, match_score) triples.
    Scores are computed via the rule-based matching engine as ground truth.
    """
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        cand_skills = skill_to_vector(r["candidate_skills"])
        cand_acc = accommodation_to_vector(r["candidate_accommodation_needs"])
        cand_dis = disability_to_vector(r["disability_type"])
        cand_feat = torch.cat([cand_skills, cand_acc, cand_dis])

        job_skills = skill_to_vector(r["job_required_skills"])
        job_acc = accommodation_to_vector(r["job_accommodations_provided"])
        job_feat = torch.cat([job_skills, job_acc, torch.zeros(len(ALL_DISABILITIES))])

        score = torch.tensor([r["match_score"]], dtype=torch.float32)
        return cand_feat, job_feat, score


# ─── Model ────────────────────────────────────────────────────────────────────

FEATURE_DIM = len(ALL_SKILLS) + len(ALL_ACCOMMODATIONS) + len(ALL_DISABILITIES)


class SkillMatchingModel(nn.Module):
    """
    Siamese-style neural network for candidate-job compatibility scoring.
    Encodes candidate and job separately, then scores their compatibility.
    """
    def __init__(self, feature_dim: int = FEATURE_DIM, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.compatibility_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, cand_feat: torch.Tensor, job_feat: torch.Tensor) -> torch.Tensor:
        cand_emb = self.encoder(cand_feat)
        job_emb = self.encoder(job_feat)
        combined = torch.cat([cand_emb, job_emb], dim=-1)
        return self.compatibility_head(combined)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)


# ─── Training ─────────────────────────────────────────────────────────────────

def generate_training_data() -> list[dict]:
    """Generate synthetic training data from seed candidates/jobs using rule-based scores."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from samaarthya_ops_env.data import CANDIDATES, JOBS
    from samaarthya_ops_env.matching_engine import compute_match_score

    records = []
    for candidate in CANDIDATES:
        for job in JOBS:
            score = compute_match_score(candidate, job)
            records.append({
                "candidate_id": candidate.id,
                "job_id": job.id,
                "candidate_skills": candidate.skills,
                "candidate_accommodation_needs": candidate.accommodation_needs,
                "disability_type": candidate.disability_type,
                "job_required_skills": job.required_skills,
                "job_accommodations_provided": job.accommodations_provided,
                "match_score": score,
            })
    return records


def train_model(
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    save_path: str = "ml/skill_matching_model.pt",
) -> SkillMatchingModel:
    """Train the skill matching model on synthetic data."""
    print("Generating training data...")
    records = generate_training_data()
    print(f"Training samples: {len(records)}")

    dataset = CandidateJobDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SkillMatchingModel(feature_dim=FEATURE_DIM)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for cand_feat, job_feat, scores in loader:
            optimizer.zero_grad()
            preds = model(cand_feat, job_feat)
            loss = criterion(preds, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "feature_dim": FEATURE_DIM,
                "embed_dim": 64,
            }, save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:>3}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Model saved to: {save_path}")
    return model


def load_model(path: str = "ml/skill_matching_model.pt") -> SkillMatchingModel:
    """Load a trained model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SkillMatchingModel(
        feature_dim=checkpoint.get("feature_dim", FEATURE_DIM),
        embed_dim=checkpoint.get("embed_dim", 64),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_match_score(
    model: SkillMatchingModel,
    candidate_skills: list[str],
    candidate_accommodation_needs: list[str],
    disability_type: str,
    job_required_skills: list[str],
    job_accommodations_provided: list[str],
) -> float:
    """Predict match score using the trained PyTorch model."""
    model.eval()
    with torch.no_grad():
        cand_feat = torch.cat([
            skill_to_vector(candidate_skills),
            accommodation_to_vector(candidate_accommodation_needs),
            disability_to_vector(disability_type),
        ]).unsqueeze(0)
        job_feat = torch.cat([
            skill_to_vector(job_required_skills),
            accommodation_to_vector(job_accommodations_provided),
            torch.zeros(len(ALL_DISABILITIES)),
        ]).unsqueeze(0)
        score = model(cand_feat, job_feat)
        return round(score.item(), 4)


if __name__ == "__main__":
    os.makedirs("ml", exist_ok=True)
    model = train_model(epochs=50, save_path="ml/skill_matching_model.pt")
    print("\nEvaluating on a sample pair...")
    score = predict_match_score(
        model,
        candidate_skills=["Coding", "JavaScript", "React", "Git"],
        candidate_accommodation_needs=["Wheelchair access", "Ergonomic workstation"],
        disability_type="Locomotor Disability",
        job_required_skills=["BPO", "Customer support", "English communication"],
        job_accommodations_provided=["Screen reader", "Flexible hours", "Wheelchair access"],
    )
    print(f"Predicted match score: {score:.4f}")
