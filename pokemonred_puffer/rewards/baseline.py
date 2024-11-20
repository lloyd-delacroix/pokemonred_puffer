import numpy as np
from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.data.events import EVENTS, REQUIRED_EVENTS
from pokemonred_puffer.data.items import REQUIRED_ITEMS, USEFUL_ITEMS
from pokemonred_puffer.data.tilesets import Tilesets
from pokemonred_puffer.data.species import StarterSpecies
from pokemonred_puffer.environment import RedGymEnv


MUSEUM_TICKET = (0xD754, 0)


class BaselineRewardEnv(RedGymEnv):
    def __init__(self, env_config: DictConfig, reward_config: DictConfig):
        super().__init__(env_config)
        self.reward_config = OmegaConf.to_object(reward_config)
        self.max_event_rew = 0
        self.starter_species = set(species.value for species in StarterSpecies)

    # TODO: make the reward weights configurable
    def get_game_state_reward(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
    
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        numBagItems = self.read_m("wNumBagItems")
        bag_item_ids = set(self.pyboy.memory[wBagItems : wBagItems + 2 * numBagItems : 2])

        return (
            {
                "event": self.reward_weight("event") * self.update_max_event_rew(),
                "seen_pokemon": self.reward_weight("seen_pokemon") * np.sum(self.seen_pokemon),
                "caught_pokemon": self.reward_weight("caught_pokemon") * np.sum(self.caught_pokemon),
                "moves_obtained": self.reward_weight("moves_obtained") * np.sum(self.moves_obtained),
                "hm_count": self.reward_weight("hm_count") * self.get_hm_count(),
                "max_party_level": self.reward_weight("max_party_level") * self.get_levels_reward(),
                "max_starter_level": self.reward_weight("max_starter_level") * self.get_starter_level_reward(),
                "first_poke_level": self.reward_weight("first_poke_level") * self.get_single_level_reward(1),
                "second_poke_level": self.reward_weight("second_poke_level") * self.get_single_level_reward(2),
                "max_opponent_level": self.reward_weight("max_opponent_level") * self.max_opponent_level,
                "death_reward": self.reward_weight("death_reward") * self.died_count,
                "blackout_check": self.reward_weight("blackout_check") * self.blackout_check,
                "badges": self.reward_weight("badges") * self.get_badges(),
                "cut_coords": self.reward_weight("cut_coords") * sum(self.cut_coords.values()),
                "cut_tiles": self.reward_weight("cut_tiles") * sum(self.cut_tiles.values()),
                "start_menu": self.reward_weight("start_menu") * self.seen_start_menu,
                "pokemon_menu": self.reward_weight("pokemon_menu") * self.seen_pokemon_menu,
                "stats_menu": self.reward_weight("stats_menu") * self.seen_stats_menu,
                "bag_menu": self.reward_weight("bag_menu") * self.seen_bag_menu,
                "seen_action_bag_menu": self.reward_weight("seen_action_bag_menu") * self.seen_action_bag_menu,
                "exploration": self.reward_weight("exploration") * np.sum(self.reward_explore_map),
                "explore_npcs": self.reward_weight("explore_npcs") * sum(self.seen_npcs.values()),
                "explore_hidden_objs": self.reward_weight("explore_hidden_objs") * sum(self.seen_hidden_objs.values()),
                "explore_signs": self.reward_weight("explore_signs") * sum(self.seen_signs.values()),
                "pokecenter_heal": self.reward_weight("pokecenter_heal") * self.pokecenter_heal,
                "a_press": self.reward_weight("a_press") * len(self.a_press),
                "warps": self.reward_weight("explore_warps") * len(self.seen_warps),
                "use_surf": self.reward_weight("use_surf") * self.use_surf,"rival3": self.reward_weight("required_event") * int(self.read_m("wSSAnne2FCurScript") == 4),
                "game_corner_rocket": self.reward_weight("required_event") * float(self.missables.get_missable("HS_GAME_CORNER_ROCKET")),
                "saffron_guard": self.reward_weight("required_event") * float(self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK")),
                "lapras": self.reward_weight("required_event") * float(self.flags.get_bit("BIT_GOT_LAPRAS")),
            }
            | {
                event: self.reward_weight("required_event") * float(self.events.get_event(event))
                for event in REQUIRED_EVENTS
            }
            | {
                item.name: self.reward_weight("required_item") * float(item.value in bag_item_ids)
                for item in REQUIRED_ITEMS
            }
            | {
                item.name: self.reward_weight("useful_item") * float(item.value in bag_item_ids)
                for item in USEFUL_ITEMS
            }
        )
    def reward_weight(self, weight_name):
        if weight_name in self.reward_config:
            return self.reward_config[weight_name]
        else:
            return 0

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            np.sum(self.events.get_events(EVENTS))
            - self.base_event_flags
            - int(
                self.events.get_event("EVENT_BOUGHT_MUSEUM_TICKET"),
            ),
            0,
        )

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4
    
    def get_single_level_reward(self, position):
        party_size = self.read_m("wPartyCount")
        if position > party_size:
            return 0
        else:
            return self.read_m(f"wPartyMon{position}Level") ** .66
        
    def get_starter_level_reward(self):
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            if self.read_m(f"wPartyMon{i+1}Species") in self.starter_species:
                self.max_starter_level = max(self.max_starter_level, self.read_m(f"wPartyMon{i+1}Level"))
        return self.max_starter_level ** .66
    

