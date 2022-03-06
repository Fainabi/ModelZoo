# This crawler will crawl the midi files from theorytab (https://www.hooktheory.com/theorytab)

using HTTP

domain = "https://www.hooktheory.com"

function retrive_artists(category)
    url = string(domain, "/theorytab/artists/", category)

    # read html 
    r = HTTP.get(url)    

    # the artists are in the form of /theorytab/artists/$(category[1])/$(artist name)
    html_str = String(r.body)
    pattern = Regex(string("(?<=/theorytab/artists/", category[1], "/).+(?=\")"))
    artists = map(findall(pattern, html_str)) do idxs
        html_str[idxs]
    end
end

function retrive_songs(artist)
    # page of specific artist
    # e.g. https://www.hooktheory.com/theorytab/artists/b/baby-base
    url = string(domain, "/theorytab/artists/", artist[1], "/", artist)

    r = HTTP.get(url)
    html_str = String(r.body)
    pattern = Regex(string("(?<=/theorytab/view/", artist, "/).+(?=\")"))
    # e.g. "/theorytab/view/baby-bash/suga-suga-ft-frankie-j" 
    songs = map(findall(pattern, html_str)) do idxs
        html_str[idxs]
    end
end

function retrive_api_url(artist, song)
    # the website updates its structure, and thus cannot directly get if from plaintext html
    # our target is to get the uuid of file, and GET it via api.hooktheory.com
    url = string(domain, "/theorytab/view/", artist, "/", song)

    # luckily, the `Open In Hookpad` link exposed that id, thus we can directly access it
    r = HTTP.get(url)
    html_str = String(r.body)
    pattern = r"idOfSong=.+(?=\")"

    # get api parameters
    # It may contains several different snippets
    api_urls = map(findall(pattern, html_str)) do idx
        api_params = html_str[idx]

        # extract id
        terms = split(api_params, r"[&|=]")
        api_idx = findfirst(isequal("idOfSong"), terms)
        string("https://api.hooktheory.com/v1/songs/public/", terms[api_idx+1])
    end    
end

"""
    retrive_dataset(sleeptime=0.5; savepath="theorytab)

This function will crawl the songs from theorytab artist-by-artist.
Retrived data will be stored in rawdata.json file in respective path.
"""
function retrive_dataset(sleeptime=0.5; savepath="theorytab")
    for category in 'a':'z'
        # the path to save data, if not existing such path, create one
        pathname = joinpath(savepath, string(category))
        mkpath(pathname)

        # get artists name
        artists = retrive_artists(category)
        for artist in artists, song in retrive_songs(artist), api_url in retrive_api_url(artist, song)
            # this url imitates request in the website page
            # api_url = string(api_url, "?fields=ID,xmlData,song,jsonData")

            # GET data from api
            r = HTTP.get(api_url)

            filepath = joinpath(pathname, artist, song)
            mkpath(filepath)

            # write data
            open(joinpath(filepath, "rawdata.json"), "w") do f
                write(f, r.body)
            end

            # sleep to passby anti-crawler
            sleep(sleeptime)
        end
        
    end
end

