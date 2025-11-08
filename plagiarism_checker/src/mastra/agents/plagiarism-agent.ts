import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
import { plagiarismTool } from '../tools/plagiarism-tool';
import { scorers } from '../scorers/plagiarism-scorer';

export const plagiarismAgent = new Agent({
  name: 'Plagiarism Checker',
  instructions: `
    You are a specialized plagiarism detection assistant. Your core function is to analyze given text and identify potential plagiarism.
    
When a user submits text, you MUST:
    1. First, use the plagiarismTool to analyze the text for originality
    2. After receiving the tool results, analyze them and provide a clear, structured response
    3. Always summarize the tool's findings in your response
    
    Your response should include:
    - A summary of whether plagiarism was detected
    - The confidence score and similarity metrics from the tool
    - Identification of any suspicious sections (if found)
    - Clear recommendations based on the analysis
    - An explanation of the limitations if reference text is not provided
    
    Important: You MUST provide a written response after using the tool. Never return an empty response.
    Present results in a clear and structured manner. Be factual and objective in your assessment.
  `,
  model: 'openai/gpt-4o-mini',
  tools: { plagiarismTool },
  // Temporarily disabled scorers - they're causing issues with empty responses
  // TODO: Re-enable scorers once agent response generation is working properly
  // scorers: {
  //   toolCallAppropriateness: {
  //     scorer: scorers.toolCallAppropriatenessScorer,
  //     sampling: {
  //       type: 'ratio',
  //       rate: 1,
  //     },
  //   },
  //   completeness: {
  //     scorer: scorers.completenessScorer,
  //     sampling: {
  //       type: 'ratio',
  //       rate: 1,
  //     },
  //   },
  //   plagiarismAccuracy: {
  //     scorer: scorers.plagiarismAccuracyScorer,
  //     sampling: {
  //       type: 'ratio',
  //       rate: 1,
  //     },
  //   },
  //   explanationQuality: {
  //     scorer: scorers.explanationQualityScorer,
  //     sampling: {
  //       type: 'ratio',
  //       rate: 1,
  //     },
  //   },
  // },
  memory: new Memory({
    storage: new LibSQLStore({
      url: 'file:../mastra.db', // path is relative to the .mastra/output directory
    }),
  }),
});
