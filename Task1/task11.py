Teams = ["ARG","MEX","POL","KSA"]
Matches = [("ARG","MEX"), ("ARG","POL"), ("ARG","KSA"),
           ("MEX","POL"), ("MEX","KSA"), ("POL","KSA")]

standings = {}
for team in Teams:
    standings[team] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}


def process_match(standings, team1, team2, team1_goals, team2_goals):
    standings[team1]['P']+=1
    standings[team2]['P']+=1

    standings[team1]['GF']+=team1_goals
    standings[team1]['GA']+=team2_goals
    standings[team2]['GF']+=team2_goals
    standings[team2]['GA']+=team1_goals

    if team1_goals>team2_goals:
        standings[team1]['W']+=1
        standings[team2]['L']+=1
        standings[team1]['Pts']+=3
    elif team1_goals<team2_goals:
        standings[team2]['W']+=1
        standings[team1]['L']+=1
        standings[team2]['Pts']+=3
    else:
        standings[team1]['D']+=1
        standings[team2]['D']+=1
        standings[team1]['Pts']+=1
        standings[team2]['Pts']+=1

    standings[team1]['GD']=standings[team1]['GF']-standings[team1]['GA']
    standings[team2]['GD']=standings[team2]['GF']-standings[team2]['GA']


def get_valid_score(team1, team2):
    while True:
        try:
            score_input = input(f"Enter the score for {team1} vs {team2} (format: 2-0): ")
            team1_goals, team2_goals = map(int, score_input.split("-"))
            if team1_goals < 0 or team2_goals < 0:
                print("Scores must be non-negative integers. Please try again.")
                continue
            return team1_goals, team2_goals
        except ValueError:
            print("Invalid input. Please enter the score in the format X-Y, e.g. 2-0.")


def head_to_head_result(team_a, team_b, match_history):
    # match_history stores each match's teams + goals as played
    if (team_a, team_b) in match_history:
        if match_history[(team_a, team_b)] > match_history[(team_b, team_a)]:
            return team_a
        elif match_history[(team_a, team_b)] < match_history[(team_b, team_a)]:
            return team_b
    return None


def sort_key_group(list_of_teams, standings, match_history):
    # Sort by pts, gd, gf
    sorted_teams = sorted(
        list_of_teams,
        key=lambda team: (standings[team]['Pts'], standings[team]['GD'], standings[team]['GF']),
        reverse=True
    )

    i = 0
    while i < len(sorted_teams) - 1:
        # Find the full block of teams tied with sorted_teams[i]
        tied_block = [sorted_teams[i]]
        j = i + 1
        while j < len(sorted_teams) and (
            standings[sorted_teams[i]]['Pts'] == standings[sorted_teams[j]]['Pts'] and
            standings[sorted_teams[i]]['GD'] == standings[sorted_teams[j]]['GD'] and
            standings[sorted_teams[i]]['GF'] == standings[sorted_teams[j]]['GF']
        ):
            tied_block.append(sorted_teams[j])
            j += 1

        if len(tied_block) == 2:
            # Exactly two teams tied -> apply head-to-head rule
            team_a, team_b = tied_block
            winner = head_to_head_result(team_a, team_b, match_history)
            if winner == team_b:
                sorted_teams[i], sorted_teams[i + 1] = sorted_teams[i + 1], sorted_teams[i]
        # if only 1 team (no tie) or 3+ teams tied, leave order as-is

        i = j if j > i + 1 else i + 1

    return sorted_teams


def format_gd(value):
    if value > 0:
        return f"+{value}"
    elif value < 0:
        return str(value)   # int already includes the "-" sign
    else:
        return "0"


def print_standings(standings, match_history):
    sorted_teams = sort_key_group(list(standings.keys()), standings, match_history)
    print(f"{'Team':<5} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<3} {'GA':<3} {'GD':<5} {'Pts':<4}")
    for team in sorted_teams:
        stats = standings[team]
        gd_display = format_gd(stats['GD'])
        print(f"{team:<5} {stats['P']:<3} {stats['W']:<3} {stats['D']:<3} {stats['L']:<3} "
              f"{stats['GF']:<3} {stats['GA']:<3} {gd_display:<5} {stats['Pts']:<4}")


def main():
    match_history = {}
    for team1, team2 in Matches:
        team1_goals, team2_goals = get_valid_score(team1, team2)
        process_match(standings, team1, team2, team1_goals, team2_goals)
        match_history[(team1, team2)] = team1_goals
        match_history[(team2, team1)] = team2_goals
    print_standings(standings, match_history)


if __name__ == "__main__":
    main()