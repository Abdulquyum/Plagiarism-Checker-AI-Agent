import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

// Helper function to calculate cosine similarity between two text vectors
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Create n-grams from text
function createNGrams(text: string, n: number = 3): Map<string, number> {
  const ngrams = new Map<string, number>();
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  
  for (let i = 0; i <= words.length - n; i++) {
    const ngram = words.slice(i, i + n).join(' ');
    ngrams.set(ngram, (ngrams.get(ngram) || 0) + 1);
  }
  
  return ngrams;
}

// Calculate text similarity using multiple methods
function calculateTextSimilarity(text1: string, text2: string): {
  cosineSimilarity: number;
  jaccardSimilarity: number;
  ngramOverlap: number;
  overallScore: number;
} {
  // Create n-grams
  const ngrams1 = createNGrams(text1, 3);
  const ngrams2 = createNGrams(text2, 3);
  
  // Get all unique n-grams
  const allNgrams = new Set([...ngrams1.keys(), ...ngrams2.keys()]);
  
  // Create vectors for cosine similarity
  const vec1: number[] = [];
  const vec2: number[] = [];
  
  for (const ngram of allNgrams) {
    vec1.push(ngrams1.get(ngram) || 0);
    vec2.push(ngrams2.get(ngram) || 0);
  }
  
  // Calculate cosine similarity
  const cosine = cosineSimilarity(vec1, vec2);
  
  // Calculate Jaccard similarity (intersection over union)
  let intersection = 0;
  let union = 0;
  
  for (const ngram of allNgrams) {
    const count1 = ngrams1.get(ngram) || 0;
    const count2 = ngrams2.get(ngram) || 0;
    intersection += Math.min(count1, count2);
    union += Math.max(count1, count2);
  }
  
  const jaccard = union > 0 ? intersection / union : 0;
  
  // Calculate n-gram overlap percentage
  let matchingNgrams = 0;
  for (const ngram of ngrams1.keys()) {
    if (ngrams2.has(ngram)) {
      matchingNgrams++;
    }
  }
  const ngramOverlap = ngrams1.size > 0 ? matchingNgrams / ngrams1.size : 0;
  
  // Overall score (weighted average)
  const overallScore = (cosine * 0.4 + jaccard * 0.3 + ngramOverlap * 0.3);
  
  return {
    cosineSimilarity: cosine,
    jaccardSimilarity: jaccard,
    ngramOverlap: ngramOverlap,
    overallScore: overallScore,
  };
}

// Find potentially plagiarized sections
function findSuspiciousSections(
  text: string,
  referenceText: string,
  threshold: number = 0.7
): Array<{
  text: string;
  similarity: number;
  startIndex: number;
  endIndex: number;
}> {
  const suspiciousSections: Array<{
    text: string;
    similarity: number;
    startIndex: number;
    endIndex: number;
  }> = [];
  
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const refWords = referenceText.toLowerCase().split(/\s+/);
  
  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    const words = sentence.toLowerCase().split(/\s+/);
    
    // Check if sentence has significant overlap with reference
    if (words.length < 5) continue; // Skip very short sentences
    
    // Find best matching section in reference
    let maxSimilarity = 0;
    let bestMatch = '';
    
    for (let j = 0; j <= refWords.length - words.length; j++) {
      const refSection = refWords.slice(j, j + words.length).join(' ');
      const similarity = calculateTextSimilarity(sentence, refSection).overallScore;
      
      if (similarity > maxSimilarity) {
        maxSimilarity = similarity;
        bestMatch = refSection;
      }
    }
    
    if (maxSimilarity >= threshold) {
      const startIndex = text.indexOf(sentence);
      suspiciousSections.push({
        text: sentence.trim(),
        similarity: maxSimilarity,
        startIndex: startIndex,
        endIndex: startIndex + sentence.length,
      });
    }
  }
  
  return suspiciousSections;
}

export const plagiarismTool = createTool({
  id: 'check-plagiarism',
  description: 'Analyze text for potential plagiarism by comparing it with reference text or checking for similarity patterns',
  inputSchema: z.object({
    text: z.string().describe('The text to check for plagiarism'),
    referenceText: z.string().optional().describe('Optional reference text to compare against'),
    threshold: z.number().min(0).max(1).default(0.7).describe('Similarity threshold (0-1) for flagging potential plagiarism'),
  }),
  outputSchema: z.object({
    isLikelyPlagiarized: z.boolean().describe('Whether the text is likely plagiarized'),
    confidenceScore: z.number().min(0).max(1).describe('Confidence score (0-1) for the plagiarism assessment'),
    overallSimilarity: z.number().min(0).max(1).describe('Overall similarity score if reference text is provided'),
    suspiciousSections: z.array(z.object({
      text: z.string(),
      similarity: z.number(),
      startIndex: z.number(),
      endIndex: z.number(),
    })).describe('Sections of text that may be plagiarized'),
    analysis: z.object({
      cosineSimilarity: z.number().optional(),
      jaccardSimilarity: z.number().optional(),
      ngramOverlap: z.number().optional(),
    }).optional(),
    recommendations: z.array(z.string()).describe('Recommendations based on the analysis'),
  }),
  execute: async ({ context }) => {
    const { text, referenceText, threshold } = context;
    
    if (!text || text.trim().length === 0) {
      throw new Error('Text to analyze cannot be empty');
    }
    
    // If reference text is provided, compare directly
    if (referenceText && referenceText.trim().length > 0) {
      const similarity = calculateTextSimilarity(text, referenceText);
      const suspiciousSections = findSuspiciousSections(text, referenceText, threshold);
      const isLikelyPlagiarized = similarity.overallScore >= threshold;
      
      const recommendations: string[] = [];
      if (isLikelyPlagiarized) {
        recommendations.push('High similarity detected. Consider rewriting the identified sections.');
        recommendations.push('Ensure proper citation if this is a reference to existing work.');
      } else if (similarity.overallScore >= threshold * 0.8) {
        recommendations.push('Moderate similarity detected. Review for proper attribution.');
      } else {
        recommendations.push('Text appears original. Continue to ensure proper citation of any sources.');
      }
      
      return {
        isLikelyPlagiarized,
        confidenceScore: similarity.overallScore,
        overallSimilarity: similarity.overallScore,
        suspiciousSections,
        analysis: {
          cosineSimilarity: similarity.cosineSimilarity,
          jaccardSimilarity: similarity.jaccardSimilarity,
          ngramOverlap: similarity.ngramOverlap,
        },
        recommendations,
      };
    }
    
    // Without reference text, perform pattern-based analysis
    // Check for common plagiarism indicators
    const words = text.split(/\s+/);
    const uniqueWords = new Set(words.map(w => w.toLowerCase()));
    const diversityRatio = uniqueWords.size / words.length;
    
    // Check for unusual patterns that might indicate copying
    const suspiciousPatterns = [
      /(?:citation|source|reference)\s*:\s*\d+/gi, // Citation patterns
      /\[.*?\]/g, // Reference brackets
    ];
    
    let patternMatches = 0;
    for (const pattern of suspiciousPatterns) {
      const matches = text.match(pattern);
      if (matches) patternMatches += matches.length;
    }
    
    // Low diversity might indicate copied text
    const isLikelyPlagiarized = diversityRatio < 0.3;
    const confidenceScore = isLikelyPlagiarized ? 0.6 : 0.3;
    
    const recommendations: string[] = [];
    if (isLikelyPlagiarized) {
      recommendations.push('Low word diversity detected. Consider rewriting for better originality.');
    }
    if (patternMatches > 0) {
      recommendations.push('Citation patterns detected. Ensure all sources are properly attributed.');
    }
    recommendations.push('For accurate plagiarism detection, provide reference text to compare against.');
    
    return {
      isLikelyPlagiarized,
      confidenceScore,
      overallSimilarity: 0,
      suspiciousSections: [],
      recommendations,
    };
  },
});

