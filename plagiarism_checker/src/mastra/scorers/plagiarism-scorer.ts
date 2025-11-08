import { z } from 'zod';
import { createToolCallAccuracyScorerCode } from '@mastra/evals/scorers/code';
import { createCompletenessScorer } from '@mastra/evals/scorers/code';
import { createScorer } from '@mastra/core/scores';

export const toolCallAppropriatenessScorer = createToolCallAccuracyScorerCode({
  expectedTool: 'plagiarismTool',
  strictMode: false,
});

export const completenessScorer = createCompletenessScorer();

// Custom LLM-judged scorer: evaluates if plagiarism detection is accurate and thorough
export const plagiarismAccuracyScorer = createScorer({
  name: 'Plagiarism Detection Accuracy',
  description:
    'Checks that the plagiarism detection results are accurate and the analysis is thorough',
  type: 'agent',
  judge: {
    model: 'openai/gpt-4o-mini',
    instructions:
      'You are an expert evaluator of plagiarism detection systems. ' +
      'Evaluate whether the assistant correctly identified plagiarism, provided accurate similarity scores, ' +
      'and gave helpful recommendations. Consider both false positives and false negatives. ' +
      'Return only the structured JSON matching the provided schema.',
  },
})
  .preprocess(({ run }) => {
    const userText = (run.input?.inputMessages?.[0]?.content as string) || '';
    const assistantText = (run.output?.[0]?.content as string) || '';
    const toolCalls = run.output?.[0]?.toolCalls || [];
    const toolResults = run.output?.[0]?.toolResults || [];
    return { userText, assistantText, toolCalls, toolResults };
  })
  .analyze({
    description:
      'Extract plagiarism detection results and evaluate their accuracy',
    outputSchema: z.object({
      toolUsed: z.boolean(),
      plagiarismDetected: z.boolean(),
      confidenceScore: z.number().min(0).max(1).optional(),
      analysisProvided: z.boolean(),
      recommendationsProvided: z.boolean(),
      accuracy: z.number().min(0).max(1).default(0.5),
      explanation: z.string().default(''),
    }),
    createPrompt: ({ results }) => {
      const { userText, assistantText, toolCalls, toolResults } = results.preprocessStepResult;
      
      return `
            You are evaluating a plagiarism detection assistant's performance.
            
            User request:
            """
            ${userText}
            """
            
            Assistant response:
            """
            ${assistantText}
            """
            
            Tool calls made: ${JSON.stringify(toolCalls, null, 2)}
            Tool results: ${JSON.stringify(toolResults, null, 2)}
            
            Tasks:
            1) Check if the plagiarism tool was used appropriately
            2) Evaluate if plagiarism was correctly identified (check for false positives/negatives)
            3) Assess if confidence scores and similarity metrics are reasonable
            4) Verify that suspicious sections were identified if plagiarism was detected
            5) Check if helpful recommendations were provided
            6) Determine overall accuracy of the detection
            
            Return JSON with fields:
            {
              "toolUsed": boolean,
              "plagiarismDetected": boolean,
              "confidenceScore": number (0-1, optional),
              "analysisProvided": boolean,
              "recommendationsProvided": boolean,
              "accuracy": number (0-1),
              "explanation": string
            }
        `;
    },
  })
  .generateScore(({ results }) => {
    const r = (results as any)?.analyzeStepResult || {};
    
    if (!r.toolUsed) return 0; // Must use the tool
    
    let score = 0.5; // Base score
    
    // Increase score based on detection accuracy
    if (r.accuracy !== undefined) {
      score = r.accuracy;
    }
    
    // Bonus for providing analysis
    if (r.analysisProvided) score += 0.1;
    
    // Bonus for providing recommendations
    if (r.recommendationsProvided) score += 0.1;
    
    return Math.min(1, Math.max(0, score));
  })
  .generateReason(({ results, score }) => {
    const r = (results as any)?.analyzeStepResult || {};
    return `Plagiarism detection scoring: toolUsed=${r.toolUsed ?? false}, accuracy=${r.accuracy ?? 0}, analysisProvided=${r.analysisProvided ?? false}, recommendationsProvided=${r.recommendationsProvided ?? false}. Score=${score}. ${r.explanation ?? ''}`;
  });

// Scorer for checking if the assistant provides clear explanations
export const explanationQualityScorer = createScorer({
  name: 'Explanation Quality',
  description:
    'Evaluates whether the assistant provides clear and helpful explanations about plagiarism detection results',
  type: 'agent',
  judge: {
    model: 'openai/gpt-4o-mini',
    instructions:
      'You are an expert evaluator of technical explanations. ' +
      'Evaluate whether the assistant provided clear, understandable explanations about plagiarism detection results. ' +
      'Return only the structured JSON matching the provided schema.',
  },
})
  .preprocess(({ run }) => {
    const assistantText = (run.output?.[0]?.content as string) || '';
    return { assistantText };
  })
  .analyze({
    description: 'Evaluate the quality and clarity of the explanation',
    outputSchema: z.object({
      clarity: z.number().min(0).max(1).describe('How clear is the explanation'),
      completeness: z.number().min(0).max(1).describe('How complete is the explanation'),
      helpfulness: z.number().min(0).max(1).describe('How helpful is the explanation'),
      explanation: z.string().default(''),
    }),
    createPrompt: ({ results }) => {
      return `
            Evaluate the quality of this plagiarism detection explanation:
            """
            ${results.preprocessStepResult.assistantText}
            """
            
            Rate the following aspects (0-1):
            - Clarity: Is the explanation easy to understand?
            - Completeness: Does it cover all important aspects?
            - Helpfulness: Does it provide actionable insights?
            
            Return JSON:
            {
              "clarity": number,
              "completeness": number,
              "helpfulness": number,
              "explanation": string
            }
        `;
    },
  })
  .generateScore(({ results }) => {
    const r = (results as any)?.analyzeStepResult || {};
    const avgScore = (r.clarity + r.completeness + r.helpfulness) / 3;
    return isNaN(avgScore) ? 0 : Math.max(0, Math.min(1, avgScore));
  })
  .generateReason(({ results, score }) => {
    const r = (results as any)?.analyzeStepResult || {};
    return `Explanation quality: clarity=${r.clarity ?? 0}, completeness=${r.completeness ?? 0}, helpfulness=${r.helpfulness ?? 0}. Score=${score}. ${r.explanation ?? ''}`;
  });

export const scorers = {
  toolCallAppropriatenessScorer,
  completenessScorer,
  plagiarismAccuracyScorer,
  explanationQualityScorer,
};

