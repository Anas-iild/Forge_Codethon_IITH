import { Database } from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import BetterSqlite3 from "better-sqlite3";
import { eq } from "drizzle-orm";
import {
  candidates, signalData, predictions,
  type Candidate, type SignalDataPoint, type Prediction,
  type InsertCandidate, type InsertSignalData, type InsertPrediction,
} from "@shared/schema";

const sqlite = new BetterSqlite3("database.sqlite");
const db = drizzle(sqlite);

// Run migrations
sqlite.exec(`
  CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    current_role TEXT NOT NULL,
    department TEXT NOT NULL,
    years_experience REAL NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS signal_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    month INTEGER NOT NULL,
    performance REAL NOT NULL,
    skill_growth REAL NOT NULL,
    collaboration REAL NOT NULL,
    initiative REAL NOT NULL,
    delivery REAL NOT NULL
  );

  CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    predicted_role TEXT NOT NULL,
    timeframe_months INTEGER NOT NULL,
    confidence_score REAL NOT NULL,
    growth_velocity REAL NOT NULL,
    learning_velocity REAL NOT NULL,
    role_fit_score REAL NOT NULL,
    summary TEXT NOT NULL,
    strengths TEXT NOT NULL,
    risk_factors TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
`);

export interface IStorage {
  // Candidates
  getAllCandidates(): Candidate[];
  getCandidateById(id: number): Candidate | undefined;
  createCandidate(data: InsertCandidate): Candidate;
  updateCandidate(id: number, data: Partial<InsertCandidate>): Candidate | undefined;
  deleteCandidate(id: number): void;

  // Signal data
  getSignalDataByCandidate(candidateId: number): SignalDataPoint[];
  upsertSignalData(data: InsertSignalData[]): void;
  deleteSignalData(candidateId: number): void;

  // Predictions
  getPredictionByCandidate(candidateId: number): Prediction | undefined;
  savePrediction(data: InsertPrediction): Prediction;
  getAllPredictions(): Prediction[];
}

export class SqliteStorage implements IStorage {
  getAllCandidates(): Candidate[] {
    return db.select().from(candidates).all();
  }

  getCandidateById(id: number): Candidate | undefined {
    return db.select().from(candidates).where(eq(candidates.id, id)).get();
  }

  createCandidate(data: InsertCandidate): Candidate {
    return db.insert(candidates).values(data).returning().get();
  }

  updateCandidate(id: number, data: Partial<InsertCandidate>): Candidate | undefined {
    return db.update(candidates).set(data).where(eq(candidates.id, id)).returning().get();
  }

  deleteCandidate(id: number): void {
    db.delete(signalData).where(eq(signalData.candidateId, id)).run();
    db.delete(predictions).where(eq(predictions.candidateId, id)).run();
    db.delete(candidates).where(eq(candidates.id, id)).run();
  }

  getSignalDataByCandidate(candidateId: number): SignalDataPoint[] {
    return db.select().from(signalData)
      .where(eq(signalData.candidateId, candidateId))
      .all()
      .sort((a, b) => a.month - b.month);
  }

  upsertSignalData(dataArr: InsertSignalData[]): void {
    for (const d of dataArr) {
      db.insert(signalData).values(d).run();
    }
  }

  deleteSignalData(candidateId: number): void {
    db.delete(signalData).where(eq(signalData.candidateId, candidateId)).run();
  }

  getPredictionByCandidate(candidateId: number): Prediction | undefined {
    return db.select().from(predictions)
      .where(eq(predictions.candidateId, candidateId))
      .get();
  }

  savePrediction(data: InsertPrediction): Prediction {
    // Delete existing prediction for this candidate first
    db.delete(predictions).where(eq(predictions.candidateId, data.candidateId)).run();
    return db.insert(predictions).values(data).returning().get();
  }

  getAllPredictions(): Prediction[] {
    return db.select().from(predictions).all();
  }
}

export const storage = new SqliteStorage();
