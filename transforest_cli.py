#!/usr/bin/env python3

import transforest
from groq import Groq
import openai
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import sys

console = Console()

# Provider configurations
PROVIDERS = {
    "groq": {
        "name": "Groq",
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    },
    "openai": {
        "name": "OpenAI", 
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5"]
    }
}

DECORATORS = {
    "mbr": {"name": "Minimum Bayes Risk", "desc": "Lowest cosine distance"},
    "voting": {"name": "Majority Voting", "desc": "Most frequent response"}
}

def create_llm_function(provider, model, api_key, decorator_type, num_calls):
    """Create a dynamically decorated LLM function"""
    
    if decorator_type == "mbr":
        decorator = transforest.minimum_bayes_risk(num_calls=num_calls)
    else:
        decorator = transforest.majority_voting(num_calls=num_calls)
    
    if provider == "groq":
        client = Groq(api_key=api_key)
        
        @decorator
        def ask_llm(prompt):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"
                
    elif provider == "openai":
        client = openai.OpenAI(api_key=api_key)
        
        @decorator
        def ask_llm(prompt):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"
    
    return ask_llm

def display_results(result, decorator_type):
    """Display results with minimal formatting"""
    console.print(f"\n[bold green]Selected:[/bold green] {result['selected_response']}")
    console.print(f"[dim]Time: {result['total_execution_time']:.2f}s | Index: {result['selected_index']}[/dim]")
    
    if decorator_type == "mbr":
        console.print(f"[dim]Best distance: {min(resp['avg_distance'] for resp in result['all_responses']):.4f}[/dim]")
    else:
        max_votes = max(result['vote_counts'].values())
        console.print(f"[dim]Votes: {max_votes}/{len(result['all_responses'])}[/dim]")

def display_detailed_results(result, decorator_type):
    """Display detailed results with all responses, times, and data"""
    console.print(f"\n[bold]DETAILS[/bold]")
    console.print(f"Selected: {result['selected_response']}")
    console.print(f"Time: {result['total_execution_time']:.2f}s | Calls: {len(result['all_responses'])} | Index: {result['selected_index']}")
    
    from rich.table import Table
    
    if decorator_type == "mbr":
        console.print(f"Best distance: {min(resp['avg_distance'] for resp in result['all_responses']):.4f}\n")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Response", max_width=60)
        table.add_column("Time", width=6)
        table.add_column("Distance", width=8)
        
        for resp in result['all_responses']:
            is_best = resp['index'] == result['selected_index']
            marker = "*" if is_best else " "
            table.add_row(
                f"{marker}{resp['index']}",
                resp['response'],
                f"{resp['execution_time']:.2f}s",
                f"{resp['avg_distance']:.4f}",
                style="bold green" if is_best else None
            )
        
    else:  # voting
        max_votes = max(result['vote_counts'].values())
        console.print(f"Max votes: {max_votes}/{len(result['all_responses'])}\n")
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Response", max_width=60)
        table.add_column("Time", width=6)
        table.add_column("Votes", width=6)
        
        for resp in result['all_responses']:
            is_best = resp['index'] == result['selected_index']
            marker = "*" if is_best else " "
            table.add_row(
                f"{marker}{resp['index']}",
                resp['response'],
                f"{resp['execution_time']:.2f}s",
                str(resp['vote_count']),
                style="bold green" if is_best else None
            )
    
    console.print(table)


def show_welcome():
    """Show welcome message"""
    console.clear()
    console.print("[bold blue]Transforest CLI[/bold blue]")
    console.print("[dim]Commands: q=quit, b=back, d=details (after results)[/dim]\n")

def select_provider():
    """Select LLM provider"""
    console.print("[bold]Select Provider:[/bold]")
    for i, (key, provider) in enumerate(PROVIDERS.items(), 1):
        console.print(f"  {i}. {provider['name']}")
    
    choice = Prompt.ask(
        "Choice",
        choices=[str(i) for i in range(1, len(PROVIDERS) + 1)] + ["q", "b"],
        default="1"
    )
    
    if choice == "q":
        sys.exit(0)
    elif choice == "b":
        return None
    else:
        provider_keys = list(PROVIDERS.keys())
        return provider_keys[int(choice) - 1]

def select_model(provider):
    """Select model for chosen provider"""
    provider_info = PROVIDERS[provider]
    console.print(f"\n[bold]Select {provider_info['name']} Model:[/bold]")
    for i, model in enumerate(provider_info["models"], 1):
        console.print(f"  {i}. {model}")
    
    choice = Prompt.ask(
        "Choice",
        choices=[str(i) for i in range(1, len(provider_info["models"]) + 1)] + ["q", "b"],
        default="1"
    )
    
    if choice == "q":
        sys.exit(0)
    elif choice == "b":
        return None
    else:
        return provider_info["models"][int(choice) - 1]

def get_api_key(provider):
    """Get API key input"""
    provider_name = PROVIDERS[provider]["name"]
    console.print(f"\n[bold]Enter {provider_name} API Key:[/bold]")
    
    while True:
        api_key = Prompt.ask("API Key", password=True)
        
        if api_key.lower() == "q":
            sys.exit(0)
        elif api_key.lower() == "b":
            return None
        elif api_key.strip():
            return api_key.strip()
        else:
            console.print("[red]API Key cannot be empty[/red]")

def select_decorator():
    """Select decorator type"""
    console.print("\n[bold]Select Decorator:[/bold]")
    for i, (key, decorator) in enumerate(DECORATORS.items(), 1):
        console.print(f"  {i}. {decorator['name']} - {decorator['desc']}")
    
    choice = Prompt.ask(
        "Choice",
        choices=[str(i) for i in range(1, len(DECORATORS) + 1)] + ["q", "b"],
        default="1"
    )
    
    if choice == "q":
        sys.exit(0)
    elif choice == "b":
        return None
    else:
        decorator_keys = list(DECORATORS.keys())
        return decorator_keys[int(choice) - 1]

def select_num_calls():
    """Select number of calls"""
    console.print("\n[bold]Number of calls (1-20):[/bold]")
    
    while True:
        try:
            num_input = Prompt.ask("Number", default="5")
            
            if num_input.lower() == "q":
                sys.exit(0)
            elif num_input.lower() == "b":
                return None
            
            num = int(num_input)
            if 1 <= num <= 20:
                return num
            else:
                console.print("[red]Number must be between 1 and 20[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")

def get_prompt():
    """Get user prompt input"""
    console.print("\n[bold]Enter your question:[/bold]")
    
    while True:
        prompt = Prompt.ask("Prompt")
        
        if prompt.lower() == "q":
            sys.exit(0)
        elif prompt.lower() == "b":
            return None
        elif prompt.strip():
            return prompt.strip()
        else:
            console.print("[red]Prompt cannot be empty[/red]")

def show_config(provider, model, decorator_type, num_calls):
    """Show final configuration"""
    console.print(f"\n[bold]Config:[/bold] {PROVIDERS[provider]['name']} | {model} | {DECORATORS[decorator_type]['name']} | {num_calls} calls")

def ask_question_loop(provider, model, api_key, decorator_type, num_calls):
    """Continuous question loop with saved settings"""
    while True:
        # Get prompt
        user_prompt = get_prompt()
        if user_prompt is None:
            # User pressed 'b', exit loop
            return
        
        # Execute
        show_config(provider, model, decorator_type, num_calls)
        llm_function = create_llm_function(provider, model, api_key, decorator_type, num_calls)
        
        with console.status(f"Running {DECORATORS[decorator_type]['name']}..."):
            result = llm_function(user_prompt)
        
        display_results(result, decorator_type)
        
        # Options: y/d/q
        choice = Prompt.ask(
            "\nOptions",
            choices=["y", "d", "q"],
            default="y"
        )
        
        if choice == "q":
            console.print("Goodbye! 👋")
            sys.exit(0)
        elif choice == "d":
            display_detailed_results(result, decorator_type)
            # After showing details, ask again
            continue_choice = Prompt.ask(
                "\nAnother question?",
                choices=["y", "q"],
                default="y"
            )
            if continue_choice == "q":
                console.print("Goodbye! 👋")
                sys.exit(0)
        # If y, continue loop for new question

def main():
    """Main interactive CLI function"""
    show_welcome()
    
    # Step 1: Provider selection
    provider = select_provider()
    if provider is None:
        main()
        return
    
    # Step 2: API Key input
    while True:
        api_key = get_api_key(provider)
        if api_key is None:
            # Go back to provider selection
            main()
            return
        break
    
    # Step 3: Model selection
    while True:
        model = select_model(provider)
        if model is None:
            # Go back to API key input
            continue
        break
    
    # Step 4: Decorator selection
    while True:
        decorator_type = select_decorator()
        if decorator_type is None:
            # Go back to model selection
            continue
        break
    
    # Step 5: Number of calls selection
    while True:
        num_calls = select_num_calls()
        if num_calls is None:
            # Go back to decorator selection
            continue
        break
    
    # Step 6: Enter question loop
    ask_question_loop(provider, model, api_key, decorator_type, num_calls)

if __name__ == "__main__":
    main()