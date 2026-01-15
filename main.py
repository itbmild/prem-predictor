from src.models.dataset import PremierLeagueDataset

MATCHES_DIR = './data/processed/processed_full/prem-data(2015-2025).csv'


def main():
    # test the dataloader
    dataset = PremierLeagueDataset(MATCHES_DIR)
    print(dataset.__getitem__(2))


    

if __name__ == "__main__":
    main()