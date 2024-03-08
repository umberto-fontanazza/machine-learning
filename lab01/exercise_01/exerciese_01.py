from sys import argv

def total_score(votes: list[float]) -> float:
    return sum(votes) - min(votes) - max(votes)

def main():
    filename: str = argv[1]
    country_scores: dict[str, float] = {}
    athlete_scores = []

    with open(filename, 'r') as file:
        for line in file:
            tokens = line.split(' ')
            name, surname, nationality = tokens[0:3]
            print(f'{name = }, {surname = }, {nationality = }')
            scores: list[float] = [float(token) for token in tokens[3:]]
            score = total_score(scores)
            athlete_scores.append((f'{name} {surname}', score))
            nationality_partial = country_scores.get(nationality, 0)
            country_scores[nationality] = nationality_partial + score

    athlete_scores.sort(key= lambda tup: tup[1], reverse=True)
    country_totals: list[tuple[str, float]] = [(nationality, score) for nationality, score in country_scores.items()]
    best_country, best_couhntry_score = max(country_totals, key= lambda tup: tup[1])

    print(athlete_scores[:3])
    print('Top country')
    print(f'{best_country = }, {best_couhntry_score = }')

if __name__ == '__main__':
    main()