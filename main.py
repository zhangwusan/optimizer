from optimizer4OS.cleaner.factor import CleanerFactory

def main():
    cleaner = CleanerFactory.get_cleaner()
    cleaner.find_large_files(directory="~", size_mb=1024)

if __name__ == "__main__":
    print("Starting platform cleaner...")
    main()