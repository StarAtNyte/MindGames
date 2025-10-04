In the rules, it says that "Heavy Heuristics: Use of hard-coded responses or non-generalizable solutions" are prohibited. What are examples of such heuristics? Would incorporating something like a minimax tree search be in violation of the rule? Thanks.
Kevin Wang — 8/9/2025 11:14 AM
Hi @tamaku Good questions! 
When we say “Heavy Heuristics”, we mean solutions that bypass the LLM’s own reasoning by relying on pre-programmed rules, scripts, or exhaustive lookup tables that only work for specific cases. Examples include:

Pre-writing responses for every possible game state.
Embedding deterministic “if–then” decision trees in the prompt so the LLM just follows steps without thinking.
Using a fixed state→action mapping instead of letting the LLM infer the move.

About minimax:
Prompting the LLM to simulate a minimax-style reasoning process in its own output is fine, that still requires the model to reason.
What’s not allowed is calling an external minimax tree search algorithm that computes the move entirely outside the LLM, then just feeding the answer back for output.

Rule of thumb: If the decision is computed outside the LLM and the model just parrots it, that’s heavy heuristics. If the LLM is actually doing the reasoning, it’s fine.
Also apologies for the late reply, I just saw your message😅
Akash

 — 8/9/2025 3:34 PM
I have a question.
If I have a standard default answer (not an entire algorithm) that is a fallback in case my LLM does not return the answer in the required format, is that fine?
It's not the most optimal answer, just enough to allow the model to not lose directly for not outputting in the required format
Kevin Wang — 8/10/2025 3:03 AM
Yeah that's fine.
jokrasa — 8/10/2025 6:12 AM
Hi, I have a question about this as well - in another question thread it was said we could mix LLMs with rl and rule based decision making, that made me think something like metas approach with Cicero would be fine. However, there, most of the reasoning was happening outside the llm which was mostly decoding/encoding the dialogue answers.  According  to this answer here it now seems like that wouldn’t be allowed. Can u clarify pls?
vxtor6025 — 8/14/2025 2:21 PM
++ same question from me
DanDan

 — 8/20/2025 2:32 PM
I have another question regarding fallback mechanics, specifically in the generalization track.

Currently, my agent script never checks which game it is playing (to comply with the generalization rule). However, to know what format to follow, it needs to identify the game in order to apply the specific format rules. I’m not sure whether this will be considered as using a different system for each game.

Also, the rules for fallback answers are important. For example, in the Prisoner’s Dilemma game, a badly formatted decision is treated as cooperation. It seems strange to allow manually checking the format and setting the fallback to something else (like defect).

Finally, if avoiding instant losses using a fallback answer is allowed, are we permitted to filter illegal clues in the Codenames game? Although this isn’t strictly a “format” issue, it does prevent an instant loss.
Kevin Wang — 8/20/2025 10:56 PM
@DanDan That’s a great point. We’re having an internal meeting tonight to go over these issues in detail, and we’ll share an update with you by tomorrow.
Kevin Wang — 8/20/2025 10:59 PM
@jokrasa @vxtor6025 Yes, the reasoning should take place within the LLMs themselves, or be handled by the main agentic model.😃
DanDan

 — 8/25/2025 8:32 AM
Is there any update to this?
Kevin Wang — 8/25/2025 12:59 PM
We are leaning to prohibiting fallback in this cases. But we need further confirmation from the team, since this is a change from previous rules. Really appreciate you brought up this point
Kevin Wang — 8/27/2025 4:15 AM
Hi @DanDan 

To address recent concerns and ensure fairness in the competition, we are in the process of updating the rules. We’d like to share two important updates currently being drafted (not finalized yet):

Rule Change on Fallback Answers

Previously, fallback answers were allowed. Moving forward, they will be classified as heavy heuristics.

Every response must come directly from the LLM and be reasoned by the agent. Extracting answers from the LLM is valid, but relying on fallback answers is not.

Again thank @DanDan  for raising the feedback here. We review the game data, and it showed that fallback answers created an unintended unfair advantage.

TrueSkill Progression Against Bots

We’ve received feedback that TrueSkill ratings improve too slowly when playing against bots.

To address this, we are introducing a Second Stage to the competition.

Stage 2 Details

The top 60% of teams from Stage 1 will qualify for Stage 2.
Stage 2 will open after the current competition end date,  in two 48-hour windows: October 11–12 and October 18–19.
-In Stage 2, no bots will be available—all matches will be against other teams.
To be considered for final rankings, teams must complete:
15 games in the Social Decision Track
3 × 8 games in the Generalization Track
DanDan

 — 8/27/2025 9:28 AM
Thanks for clarifying on the rules, I do think this change is better for the fairness of the competition
DanDan

 — 8/27/2025 9:30 AM
Additional question, is stage 2 going to be run by tournament official after submitting, or are we able to self-host and make additional changes to our agent. 
DanDan

 — 8/27/2025 9:39 AM
Another concern is limiting stage 2 to top 60% of teams, as there are not many teams competing for Generalization Track now, and how is this top 60% decided? 
ChiaraThöni — 8/28/2025 2:48 AM
Hi @DanDan , stage 2 will be self-hosted as well, we will not make any changes on that aspect.
ChiaraThöni — 8/28/2025 2:51 AM
You are raising a valid point here! The top 60% will be based on the models' TrueSkill rating. However, because of the small number of teams that are actively competing, we may decide to include a larger fraction of teams in stage 2.
martianlantern — 9/15/2025 2:08 PM
Hello I want to thank the organizers for hosting this competition
@Kevin Wang I understand that modifying the agent outputs based on preconditioned rules is not allowed. Does changing the input prompt based on the current phase of the game consider as "heavy heuristics" 
martianlantern — 9/15/2025 2:47 PM
doing something like this for mafia for ex
def __call__(self, observation: str) -> str:

    phase = self.extract_current_phase(observation)

    if phase == "night":
        reponse = self.model.generate(night_prompt + observation)
    elif phase == "discussion":
        response = self.model.generate(discussion_prompt + observation)
    elif phase == "vote":
        reponse = self.model.generate(vote_prompt + observation)
        
    return response
 
Kevin Wang — 9/16/2025 4:01 AM
Hi @martianlantern  good question and yeah this would be fine :D


Is it allowed that I attach special heads on top of my model for the special game phases where format adherence is strict req.

I think that it may be that we may hinder the model thinking by threatening it to abide the output format... Like : Torturing model not to spit extra or less as per the format... which can force the model to just fit the format and give up or fade out  it's reasoning about the game and the current game phases...

It would be hard for model to figure out why it was beaten badly by reward ...

So these special heads (like classification head) are not the fallback but safer way for model to express it's reasoning 
Kevin Wang — Yesterday at 10:43 PM
Yeah this is allowed, ref to ⁠Clarification on Heavy Heuristi…