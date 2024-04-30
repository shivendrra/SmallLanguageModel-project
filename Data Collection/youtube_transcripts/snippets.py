"""
  --> Contains sample scripts for the data collection
  --> Labelled names
"""

from typing import Any


class SampleSnippets:
  def __init__(self) -> None:
    self.channel_ids = [
    "UCb_MAhL8Thb3HJ_wPkH3gcw", #phil edwards
    "UCA295QVkf9O1RQ8_-s3FVXg", #aevy tv
    "UCpFFItkfZz1qz5PpHpqzYBw", #nexpo
    "UCY1kMZp36IQSyNx_9h4mpCg", #mark robber
    "UCA19mAJURyYHbJzhfpqhpCA", #action lab shorts
    "UCddiUEpeqJcYeBxX1IVBKvQ", #the verge
    "UCcefcZRL2oaA_uBNeo5UOWg", #y-combinator
    "UCsQoiOrh7jzKmE8NBofhTnQ", #varun mayya
    "UCLXo7UDZvByw2ixzpQCufnA", #vox
    "UCUyvQV2JsICeLZP4c_h40kA", #thomas flight
    "UCvjgXvBlbQiydffZU7m1_aw", #the coding train
    "UCRI00CwLZdLRCWg5BdDOsNw", #canadian lad
    "UCEIwxahdLz7bap-VDs9h35A", #steve mould
    "UC4bq21IPPbpu0Qrsl7LW0sw", #slidebean
    "UCR1IuLEqb6UEA_zQ81kwXfg", #real engineering
    "UCIlU5KDHKFSaebYviKfOidw", #newsthink
    "UCtYKe7-XbaDjpUwcU5x0bLg", #neo
    "UCBJycsmduvYEL83R_U4JriQ", #mkbdh
    "UCRcgy6GzDeccI7dkbbBna3Q", #lemmino
    "UC3_BakzLfadvFrsnClMFWmQ", #john coogan
    "UCmGSJVG3mCRXVOP4yZrU1Dw", #johnny harris
    "UCFN6lQpfY8XIRdhv9G-f4bg", #henry belcaster
    "UConJDkGk921yT9hISzFqpzw", #freethink
    "UClWTCPVi-AU9TeCN6FkGARg", #EO
    "UCqnbDFdCpuN8CMEg0VuEBqA", #new york times
    "UCyHJ94JzwY92NsBVzJ2aE3Q", #econ
    "UCTqEu1wZDBju2tHkNP1dwzQ", #earthrise
    "UC6nSFpj9HTCZ5t-N3Rm3-HA", #vsauce
    "UCX6b17PVsYBQ0ip5gyeme-Q", #crash course
    "UCONtPx56PSebXJOxbFv-2jQ", #crash course kids
    "UCZYTClx2T1of7BRZ86-8fow", #sci show
    "UCzWQYUVCpZqtN93H8RR44Qw", #seeker
    "UCYbK_tjZ2OrIZFBvU6CCMiA", #brackeys
    "UCxzC4EngIsMrPmbm6Nxvb-A", #scott manley
    "UCcabW7890RKJzL968QWEykA", #CS 50
    "UCamLstJyCa-t5gfZegxsFMw", #colin and samir
    "UC415bOPUcGSamy543abLmRA", #cleo abraham
    "UCpMcsdZf2KkAnfmxiq2MfMQ", #arvin ash
    "UCqVEHtQoXHmUCfJ-9smpTSg", #answer in progress
    "UCYO_jab_esuFRV4b17AJtAw", #3blue1brown
    "UCHnyfMqiRRG1u-2MsSQLbXA", #veritasium
    "UCsXVk37bltHxD1rDPwtNM8Q", #kurzgesagt
    "UC9RM-iSvTu1uPJb8X5yp3EQ", #wendover
    "UCZaT_X_mc0BI-djXOlfhqWQ", #vice news
    "UCMiJRAwDNSNzuYeN2uWa0pA", #mrwhosetheboss
    "UCHpw8xwDNhU9gdohEcJu4aA", #theguardian
    "UCK7tptUDHh-RYDsdxO1-5QQ", #wallstreetjournal
    "UCsooa4yRKGN_zEE8iknghZA", #ted-ed
    "UC6n8I1UDTKP1IWjQMg6_TwA", #b1m
    "UC8butISFwT-Wl7EV0hUK0BQ", #free code camp
    "UCgRQHK8Ttr1j9xCEpCAlgbQ", #variety
    "UCcIXc5mJsHVYTZR1maL5l9w", #andrew ng
    "UCEBb1b_L6zDS3xTUrIALZOw", #mit opencourseware
    "UCN0QBfKk0ZSytyX_16M11fA", #mit openlearning
    "UCBpxspUNl1Th33XbugiHJzw", #mit csail
    "UC3osNjJeuDdvyALIEP-nh0g", #behind the curtain
    "UCaSCt8s_4nfkRglWCvNSDrg", #code aesthetics
    "UCjgpFI5dU-D1-kh9H1muoxQ", #hacksmith
    "UCBa659QWEk1AI4Tg--mrJ2A", #tom scott
    "UCftwRNsjfRo08xYE31tkiyw", #wired
    "UCdBK94H6oZT2Q7l0-b0xmMg", #shortcircuit
    "UCBA9cAuPy9L5IYYXqOduIvw", #encyclopedia britannica
    "UCXjmz8dFzRJZrZY8eFiXNUQ", #nerdstalgic
    "UClZbmi9JzfnB2CEb0fG8iew", #primal space
    "UCUFoQUaVRt3MVFxqwPUMLCQ", #studio binder
    "UCgLxmJ8xER7Y7sywMN5SfWg", #underfitted
    "UCac1MisHGa0qtzf0oWlU8Zw", #unpredictable
    "UCSIvk78tK2TiviLQn4fSHaw", #up and atom
    "UCUyvQV2JsICeLZP4c_h40kA", #thomas flight
    "UCqFzWxSCi39LnW1JKFR3efg", #SNL
    "UCqFzWxSCi39LnW1JKFR3efg", #peacock
    "UCccjdJEay2hpb5scz61zY6Q", #NBC
    "UC8CX0LD98EDXl4UYX1MDCXg", #valorant
    "UC6VcWc1rAoWdBCM0JxrRQ3A", #rockstar games
    "UCSHZKyawb77ixDdsGog4iWA", #lex fridman
    "UCVHdvAX5-R8y5l9xp6nroBQ", #vergecast
    "UCTb6Oy0TXI03iEUdRMR9dnw", #stuff you should know
    "UCqoAEDirJPjEUFcF2FklnBA", #star talk
    "UCccjdJEay2hpb5scz61zY6Q", #lily singh
    "UCNVBYBxWj9dMHqKEl_V8HBQ", #comedy central
    "UCNVBYBxWj9dMHqKEl_V8HBQ", #jimmy fallon
    "UCb-vZWBeWA5Q2818JmmJiqQ", #oscars
    "UChDKyKQ59fYz3JO2fl0Z6sg", #today
    "UCupvZG-5ko_eiXAupbDfxWw", #cnn
    "UCDrLGkZTcNCshOLiKi5NtEw", #wgn-news
    "UCWOA1ZGywLbqmigxE4Qlvuw", #netflix
    "UCrM7B7SL_g1edFOnmj-SDKg", #bloomberg technology
    "UCUMZ7gohGI9HcU9VNsr2FJQ", #bloomberg originals
    "UCF9imwPMSGz4Vq1NiTWCC7g", #Paramount pictures
    "UCjmJDM5pRKbUlVIzDYYWb6g", #warner bros pics
    "UCrRttZIypNTA1Mrfwo745Sg", #paramount plus
    "UC0k238zFx-Z8xFH0sxCrPJg", #architectural digest
    "UCT9zcQNlyht7fRlcjmflRSA", #imagine dragons
    "UC0C-w0YjGpqDXGB8IHb662A", #ed sheeran
    "UCgQna2EqpzqzfBjlSmzT72w", #maneskin
    "UCeLHszkByNZtPKcaVXOCOQQ", #Post Malone
    "UCjNRJBlxvvS0UXAT2Ack-QQ", #zara larsoon
    "UC-J-KZfRV8c13fOCkhXdLiQ", #dua lipa
    "UCfM3zsQsOnfWNUppiycmBuw", #eminem
    "UCNjHgaLpdy1IMNK57pYiKiQ", #aurora
    "UCqECaJ8Gagnn7YCbPEzWH6g", #taylor swift
    "UCb2HGwORFBo94DmRx4oLzow", #one direction
    "UCi4EDAgjULwwNBHOg1aaCig", #one republic
    "UCDPM_n1atn2ijUwHd0NNRQw", #coldplay
    "UCcgqSM4YEo5vVQpqwN-MaNw", #rihanna
    "UCoUM-UJ7rirJYP8CQ0EIaHA", #bruno mars
    "UC0WP5P-ufpRfjbNrmOWwLBQ", #weekend
    "UCBVjMGOIkavEAhyqpxJ73Dw", #maroon 5
    "UCPHjpfnnGklkRBBTd0k6aHg", #avicii
    "UCmHhviensDlGQeU8Yo80zdg", #dr dre
    "UC6IBMCQ6-d7p41KHxOsq4RA", #akon
    "UCiMhD4jzUqG-IgPzUmmytRQ", #queens
    "UCB0JSO6d5ysH2Mmqz5I9rIw", #ac/dc
    "UCnEiGCE13SUI7ZvojTAVBKw", #bill gates
    ]
    
  def __call__(self):
    return self.channel_ids