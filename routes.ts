import type { Express } from "express";
import type { Server } from "http";
import { storage } from "./storage";
import { insertCandidateSchema, insertSignalDataSchema } from "@shared/schema";
import { z } from "zod";

const LSTM_SERVICE_URL = "http://localhost:5001";

export function registerRoutes(httpServer: Server, app: Express): void {
  // ── Candidates ──────────────────────────────────────────────────────────────
  app.get("/api/candidates", (_req, res) => {
    const all = storage.getAllCandidates();
    res.json(all);
  });

  app.get("/api/candidates/:id", (req, res) => {
    const id = parseInt(req.params.id);
    const c = storage.getCandidateById(id);
    if (!c) return res.status(404).json({ error: "Candidate not found" });
    res.json(c);
  });

  app.post("/api/candidates", (req, res) => {
    const parsed = insertCandidateSchema.safeParse({
      ...req.body,
      createdAt: new Date().toISOString(),
    });
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    const created = storage.createCandidate(parsed.data);
    res.status(201).json(created);
  });

  app.patch("/api/candidates/:id", (req, res) => {
    const id = parseInt(req.params.id);
    const updated = storage.updateCandidate(id, req.body);
    if (!updated) return res.status(404).json({ error: "Candidate not found" });
    res.json(updated);
  });

  app.delete("/api/candidates/:id", (req, res) => {
    storage.deleteCandidate(parseInt(req.params.id));
    res.status(204).send();
  });

  // ── Signal Data ──────────────────────────────────────────────────────────────
  app.get("/api/candidates/:id/signals", (req, res) => {
    const signals = storage.getSignalDataByCandidate(parseInt(req.params.id));
    res.json(signals);
  });

  app.post("/api/candidates/:id/signals", (req, res) => {
    const candidateId = parseInt(req.params.id);
    const c = storage.getCandidateById(candidateId);
    if (!c) return res.status(404).json({ error: "Candidate not found" });

    const { signals } = req.body as {
      signals: Array<{
        month: number;
        performance: number;
        skillGrowth: number;
        collaboration: number;
        initiative: number;
        delivery: number;
      }>;
    };

    if (!Array.isArray(signals) || signals.length === 0) {
      return res.status(400).json({ error: "signals array required" });
    }

    // Clear existing signals and replace
    storage.deleteSignalData(candidateId);
    storage.upsertSignalData(
      signals.map((s) => ({
        candidateId,
        month: s.month,
        performance: s.performance,
        skillGrowth: s.skillGrowth,
        collaboration: s.collaboration,
        initiative: s.initiative,
        delivery: s.delivery,
      }))
    );

    res.json({ saved: signals.length });
  });

  // ── Prediction ───────────────────────────────────────────────────────────────
  app.post("/api/candidates/:id/predict", async (req, res) => {
    const candidateId = parseInt(req.params.id);
    const candidate = storage.getCandidateById(candidateId);
    if (!candidate) return res.status(404).json({ error: "Candidate not found" });

    const signals = storage.getSignalDataByCandidate(candidateId);
    if (signals.length < 2) {
      return res.status(400).json({ error: "Need at least 2 months of signal data to run prediction" });
    }

    try {
      const lstmPayload = {
        signals: signals.map((s) => ({
          performance: s.performance,
          skill_growth: s.skillGrowth,
          collaboration: s.collaboration,
          initiative: s.initiative,
          delivery: s.delivery,
        })),
        current_role: candidate.currentRole,
        department: candidate.department,
        years_experience: candidate.yearsExperience,
      };

      const response = await fetch(`${LSTM_SERVICE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lstmPayload),
      });

      if (!response.ok) {
        const errText = await response.text();
        return res.status(500).json({ error: `LSTM service error: ${errText}` });
      }

      const prediction = await response.json() as {
        predicted_role: string;
        timeframe_months: number;
        confidence_score: number;
        growth_velocity: number;
        learning_velocity: number;
        role_fit_score: number;
        summary: string;
        strengths: string[];
        risk_factors: string[];
        signal_velocities: Record<string, number>;
        projected_signals_6m: Record<string, number>;
        projected_signals_12m: Record<string, number>;
      };

      const saved = storage.savePrediction({
        candidateId,
        predictedRole: prediction.predicted_role,
        timeframeMonths: prediction.timeframe_months,
        confidenceScore: prediction.confidence_score,
        growthVelocity: prediction.growth_velocity,
        learningVelocity: prediction.learning_velocity,
        roleFitScore: prediction.role_fit_score,
        summary: prediction.summary,
        strengths: JSON.stringify(prediction.strengths),
        riskFactors: JSON.stringify(prediction.risk_factors),
        createdAt: new Date().toISOString(),
      });

      res.json({
        ...saved,
        strengths: prediction.strengths,
        riskFactors: prediction.risk_factors,
        signalVelocities: prediction.signal_velocities,
        projectedSignals6m: prediction.projected_signals_6m,
        projectedSignals12m: prediction.projected_signals_12m,
      });
    } catch (err: unknown) {
      console.error("Predict error:", err);
      res.status(500).json({ error: "Failed to contact LSTM service. Make sure it is running." });
    }
  });

  app.get("/api/candidates/:id/prediction", (req, res) => {
    const p = storage.getPredictionByCandidate(parseInt(req.params.id));
    if (!p) return res.status(404).json({ error: "No prediction yet" });
    res.json({
      ...p,
      strengths: JSON.parse(p.strengths),
      riskFactors: JSON.parse(p.riskFactors),
    });
  });

  // ── Dashboard Stats ──────────────────────────────────────────────────────────
  app.get("/api/stats", (_req, res) => {
    const allCandidates = storage.getAllCandidates();
    const allPredictions = storage.getAllPredictions();

    const avgConfidence = allPredictions.length > 0
      ? allPredictions.reduce((s, p) => s + p.confidenceScore, 0) / allPredictions.length
      : 0;

    const avgGrowthVelocity = allPredictions.length > 0
      ? allPredictions.reduce((s, p) => s + p.growthVelocity, 0) / allPredictions.length
      : 0;

    const fastTrack = allPredictions.filter((p) => p.timeframeMonths <= 6).length;

    res.json({
      totalCandidates: allCandidates.length,
      totalPredictions: allPredictions.length,
      avgConfidence: Math.round(avgConfidence * 10) / 10,
      avgGrowthVelocity: Math.round(avgGrowthVelocity * 10) / 10,
      fastTrackCount: fastTrack,
    });
  });
}
