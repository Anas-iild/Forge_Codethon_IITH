import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Candidates table
export const candidates = sqliteTable("candidates", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  name: text("name").notNull(),
  currentRole: text("current_role").notNull(),
  department: text("department").notNull(),
  yearsExperience: real("years_experience").notNull(),
  notes: text("notes"),
  createdAt: text("created_at").notNull(),
});

// Time-series signal data points
export const signalData = sqliteTable("signal_data", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  candidateId: integer("candidate_id").notNull(),
  month: integer("month").notNull(), // 1-indexed month number
  performance: real("performance").notNull(),   // 0-100 scale
  skillGrowth: real("skill_growth").notNull(),  // 0-100 scale
  collaboration: real("collaboration").notNull(), // 0-100 scale
  initiative: real("initiative").notNull(),      // 0-100 scale
  delivery: real("delivery").notNull(),          // 0-100 scale
});

// Predictions
export const predictions = sqliteTable("predictions", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  candidateId: integer("candidate_id").notNull(),
  predictedRole: text("predicted_role").notNull(),
  timeframeMonths: integer("timeframe_months").notNull(),
  confidenceScore: real("confidence_score").notNull(),
  growthVelocity: real("growth_velocity").notNull(),
  learningVelocity: real("learning_velocity").notNull(),
  roleFitScore: real("role_fit_score").notNull(),
  summary: text("summary").notNull(),
  strengths: text("strengths").notNull(),    // JSON array
  riskFactors: text("risk_factors").notNull(), // JSON array
  createdAt: text("created_at").notNull(),
});

// Insert schemas
export const insertCandidateSchema = createInsertSchema(candidates).omit({ id: true });
export const insertSignalDataSchema = createInsertSchema(signalData).omit({ id: true });
export const insertPredictionSchema = createInsertSchema(predictions).omit({ id: true });

// Types
export type InsertCandidate = z.infer<typeof insertCandidateSchema>;
export type InsertSignalData = z.infer<typeof insertSignalDataSchema>;
export type InsertPrediction = z.infer<typeof insertPredictionSchema>;

export type Candidate = typeof candidates.$inferSelect;
export type SignalDataPoint = typeof signalData.$inferSelect;
export type Prediction = typeof predictions.$inferSelect;
