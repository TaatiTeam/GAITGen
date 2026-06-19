import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)


def log(log_type, content):
    # Define a mapping from log types to colors
    log_colors = {
        'info': Fore.MAGENTA,
        'warning': Fore.YELLOW,
        'error': Fore.RED,
        'debug': Fore.CYAN,
        'success': Fore.GREEN,
        'default': Fore.BLUE  # default color
    }
    
    # Get the color for the given log type, defaulting to MAGENTA if not found
    color = log_colors.get(log_type, Fore.MAGENTA)

    print(f'{color}{content}{Style.RESET_ALL}')