import random

class Player:

    def __init__(self, name, position, base_attack, base_defense, stamina=100.0, yellow_card_count=0, is_suspended=False):
        self.name = name
        self.position = position.upper()
        self.base_attack = base_attack
        self.base_defense = base_defense
        self.stamina = stamina
        self.yellow_card_count = yellow_card_count
        self.is_suspended = is_suspended
        self.effective_position = self.position


    def deplete_stamina(self, rate):
        # Stamina(t) = Max(10.0, Stamina(t-1) - Base_Decay)
        self.stamina = max(10.0, self.stamina - rate)

    def get_effective_attack(self):
        return self.base_attack * (self.stamina / 100.0)

    def get_effective_defense(self):
        return self.base_defense * (self.stamina / 100.0)


class Team:
    def __init__(self, country_name, roster=None, active_lineup=None, substitutions_remaining=5, formation="4-4-2"):
        self.country_name = country_name
        self.roster = roster if roster is not None else []
        self.active_lineup = active_lineup if active_lineup is not None else []
        self.substitutions_remaining = substitutions_remaining
        self.formation = formation

    @property
    def bench(self):

        return [p for p in self.roster if p not in self.active_lineup]

    def get_aggregate_attack(self):

        total_attack = 0
        count = 0
        for player in self.active_lineup:
            if player.effective_position in ("FORWARD", "MIDFIELDER"):
                total_attack += player.get_effective_attack()
                count += 1
        return total_attack / count if count > 0 else 0

    def get_aggregate_defense(self):
        total_defense = 0
        count = 0
        for player in self.active_lineup:
            if player.effective_position in ("DEFENDER", "GOALKEEPER"):
                total_defense += player.get_effective_defense()
                count += 1
        return total_defense / count if count > 0 else 0

    def execute_substitution(self, player_out, player_in):
        if self.substitutions_remaining > 0:
            if player_out in self.active_lineup and player_in in self.bench:
                self.active_lineup.remove(player_out)
                self.active_lineup.append(player_in)
                self.substitutions_remaining -= 1
                return True
        return False

    def set_formation(self, formation):
        for p in self.active_lineup:
            p.effective_position = p.position  # reset to real position first

        if formation == "5-3-2":
            midfielders = [p for p in self.active_lineup if p.position == "MIDFIELDER"]
            if midfielders:
                midfielders[0].effective_position = "DEFENDER"
        elif formation == "3-4-3":
            defenders = [p for p in self.active_lineup if p.position == "DEFENDER"]
            if defenders:
                defenders[0].effective_position = "MIDFIELDER"
        self.formation = formation

    def apply_discipline_incident(self, player):
        player.yellow_card_count += 1
        if player.yellow_card_count >= 2:
            player.is_suspended = True
            if player in self.active_lineup:
                self.active_lineup.remove(player)
        return player.is_suspended


class MatchEvent:

    def __init__(self, event_id, event_type, minute, team, player, outcome_text):
        self.event_id = event_id
        self.event_type = event_type
        self.minute = minute
        self.team = team
        self.player = player
        self.outcome_text = outcome_text

    def to_string(self):

        who = self.player.name if self.player is not None else "Team"
        return (f"[Minute {self.minute}] {self.event_type} for "
                f"{self.team.country_name} involving {who}: {self.outcome_text}")

    def __str__(self):
        return self.to_string()


class MatchAI:
    def __init__(self, ai_model, controlled_team, risk_tolerance=1.0, decision_log=None):
        self.ai_model = ai_model
        self.controlled_team = controlled_team
        self.risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        self.decision_log = decision_log if decision_log is not None else []

    def observe_state(self, match):
        active_players = self.controlled_team.active_lineup
        avg_stamina = (sum(p.stamina for p in active_players) / len(active_players)
                        if active_players else 0)
        return {
            "score_diff": match.home_score - match.away_score,
            "minute": match.current_minute,
            "phase": match.phase,
            "team_stamina": avg_stamina,
        }

    def decide_action(self, match):
        state = self.observe_state(match)
        return self.ai_model.predict(state)

    def apply_action(self, action, match):
        if action == "SUBSTITUTE":

            if not self.controlled_team.active_lineup or not self.controlled_team.bench:
                return
            player_out = min(self.controlled_team.active_lineup, key=lambda p: p.stamina)
            eligible_subs = [p for p in self.controlled_team.bench if p.position == player_out.position]
            if eligible_subs:
                player_in = max(eligible_subs, key=lambda p: p.base_attack + p.base_defense)
                if self.controlled_team.execute_substitution(player_out, player_in):
                    self.decision_log.append(f"Substituted {player_out.name} for {player_in.name}")

        elif action == "CHANGE_FORMATION":

            self.controlled_team.set_formation("5-3-2")
            self.decision_log.append("Changed formation to Defensive 5-3-2")

        elif action == "PUSH_ATTACK":
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.2)
            self.decision_log.append(f"Pushed attack! New risk: {self.risk_tolerance:.1f}")

        elif action == "HOLD":
            self.risk_tolerance = max(0.0, self.risk_tolerance - 0.2)
            self.decision_log.append(f"Holding position. New risk: {self.risk_tolerance:.1f}")


class Match:
    def __init__(self, home_team, away_team, enable_penalty_shootout=False, widen_variance=False):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = 0
        self.away_score = 0
        self.current_minute = 0
        self.timeline = []
        self.phase = "REGULATION"
        self.base_decay = 0.5
        self.enable_penalty_shootout = enable_penalty_shootout
        self.widen_variance = widen_variance
        self.penalty_result = None

    def run_minute_tick(self, home_risk_tolerance=0.5, away_risk_tolerance=0.5):
        if self.phase != "REGULATION":
            return

        self.current_minute += 1
        home_rate = self.base_decay * (1 + (home_risk_tolerance - 0.5) * 0.4)
        away_rate = self.base_decay * (1 + (away_risk_tolerance - 0.5) * 0.4)

        for player in self.home_team.active_lineup:
            player.deplete_stamina(home_rate)
        for player in self.away_team.active_lineup:
            player.deplete_stamina(away_rate)

        self.process_goal_attempt(self.home_team, self.away_team, home_risk_tolerance)
        self.process_goal_attempt(self.away_team, self.home_team, away_risk_tolerance)

        if self.current_minute >= 90:
            if self.home_score == self.away_score:
                if self.enable_penalty_shootout:
                    self.phase = "PENALTIES"
                    self._run_penalty_shootout()
                else:
                    self.phase = "FINISHED"
            else:
                self.phase = "FINISHED"

    def process_goal_attempt(self, attacking_team, defending_team, risk_tolerance=0.5):
        if random.random() < 0.10:
            base_attack = attacking_team.get_aggregate_attack()
            base_defense = defending_team.get_aggregate_defense()
            attack_upper = 1.25 + (risk_tolerance - 0.5) * 0.4
            attack_score = base_attack * random.uniform(0.75, attack_upper)
            defense_score = base_defense * 1.3 * random.uniform(0.80, 1.20)

            if self.widen_variance:
                attack_score *= random.uniform(0.85, 1.15)

            if attack_score > defense_score:
                if attacking_team == self.home_team:
                    self.home_score += 1
                else:
                    self.away_score += 1
                event_id = f"EVT_GOAL_{self.current_minute}"
                self.timeline.append(MatchEvent(event_id, "GOAL", self.current_minute, attacking_team, None, "Goal scored!"))

    def _run_penalty_shootout(self):
    
        home_pk, away_pk = 0, 0
        board = {"home": [], "away": []}
        round_num = 0
        while True:
            if round_num >= 5 and home_pk != away_pk:
                break
            if round_num >= 20: 
                break

            home_scored = random.random() < 0.5
            board["home"].append(home_scored)
            home_pk += int(home_scored)

            away_scored = random.random() < 0.5
            board["away"].append(away_scored)
            away_pk += int(away_scored)

            round_num += 1

        winner = self.home_team if home_pk > away_pk else self.away_team
        self.penalty_result = {
            "home_pk": home_pk,
            "away_pk": away_pk,
            "winner": winner.country_name,
            "board": board,
        }
        self.phase = "FINISHED"